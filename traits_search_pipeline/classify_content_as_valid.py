"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from typing import Iterator, Mapping, List, Dict, Any, Tuple
from queue import Empty
import asyncio
import itertools
import logging
import re
from multiprocessing import Queue

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, defer, load_only

from models import (
    DB_URI,
    BaseNorm,
    Content,
    Answer,
)

from dotenv import load_dotenv
from logging_tools import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)
from llm_tools import (
    apply_llm,
    truncate_to_token_limit,
    evaluate_validity_with_llm,
)
import psutil


DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)

load_dotenv()  # loads the OPENAI_API_KEY


def insert_answer(
    session: Session,
    keyword_query_id: int,
    link_id: int,
    answer: Dict[str, str],
    log_queue: Queue,
) -> int | None:
    """Insert a new answer if there is not already one in the db."""
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )
    logger.debug(f"this is the answer: {answer}")
    if "answer_text" in answer:
        answer_text = answer["answer_text"]
    elif "answer" in answer:
        answer_text = answer["answer"]
    else:
        raise ValueError(f"expected some sort of answer key but got {answer}")
    exists_stmt = select(Answer.id).where(
        Answer.keyword_query_id == keyword_query_id,
        Answer.link_id == link_id,
        Answer.answer_text == answer_text,
    )
    existing_id = session.execute(exists_stmt).scalar_one_or_none()
    if existing_id is not None:
        logger.info(f"statement exists at {existing_id}")
        return existing_id
    try:
        ans = Answer(
            keyword_query_id=keyword_query_id,
            link_id=link_id,
            answer_text=answer_text,
            reason=answer["reason"],
            evidence=answer["evidence"],
        )
    except KeyError:
        logger.exception(f"keyerror on this answer: {answer}")
        raise
    session.add(ans)
    session.flush()
    logger.info(f"statement inserted at {ans.id}")
    return ans.id


async def call_llm_async(
    question_text: str, content_text: str, log_queue
) -> Dict[str, Any]:
    """Wrapper to sync LLM call in a thread."""
    safe_text = truncate_to_token_limit(content_text)
    return await asyncio.to_thread(
        apply_llm, question_text, safe_text, log_queue
    )


async def _worker(
    row: Dict[str, Any], sem: asyncio.Semaphore, log_queue: Queue
) -> Tuple[Dict[str, Any], str | Dict[str, Any] | None]:
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )

    async with sem:
        qtext: str = row["question_text"]
        content_text: str = row["content_text"] or ""
        logger.info(
            f"\n[QUESTION #{row['question_id']}] "
            f"(KeywordQuery #{row['keyword_query_id']}, Link #{row['link_id']})\n"
            f"Q: {qtext.strip()[:120]}{'...' if len(qtext) > 120 else ''}\n"
            f"Content: {content_text.strip()[:200]}{'...' if len(content_text) > 200 else ''}\n"
            f"{'-'*80}"
        )

        m = re.search(r"\[\[(.*?)\]\]", content_text, flags=re.DOTALL)
        if m:
            err_text = m.group(1).strip()
            answer = {
                "answer": "unknown",
                "reason": "page_error",
                "evidence": [f"[[{err_text}]]"],
            }
            return row, answer

        try:
            answer = await call_llm_async(qtext, content_text, log_queue)
            return row, answer
        except asyncio.CancelledError:
            logger.info(
                f'task cancelled for question_id={row["question_id"]}, link_id={row["link_id"]}'
            )
            raise
        except Exception as e:
            logger.exception(
                f'LLM error on question_id={row["question_id"]}, link_id={row["link_id"]}: {e}'
            )
            raise


def _get_content_text(
    session: Session, question_context: Dict[str, str]
) -> str | None:

    result = dict(question_context)
    result["content_text"] = session.execute(
        select(Content.text).where(Content.id == question_context["content_id"])
    ).scalar_one_or_none()
    return result


async def process_unanswered_questions(
    max_concurrency: int = 1,
    batch_commit: int = 50,
) -> None:
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    log_queue, listener, manager = start_process_safe_logging(
        "logs/iterate_content_for_llm.log"
    )

    main_logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(main_logger)

    engine = create_engine(DB_URI)

    main_logger.info("snapshot rows")
    with Session(engine) as session:
        unanswered_question_contexts = list(
            itertools.islice(iter_unanswered_question_contexts(session), None)
        )
        main_logger.info(unanswered_question_contexts)
        sem = asyncio.Semaphore(max_concurrency)
        main_logger.info("creating tasks")
        tasks = [
            asyncio.create_task(
                _worker(_get_content_text(session, question), sem, log_queue)
            )
            for question in unanswered_question_contexts
        ]

    pending_inserts: List[Tuple[int, int, Dict[str, Any]]] = []
    inserted = 0

    try:
        for fut in asyncio.as_completed(tasks):
            try:
                row, answer = await fut
            except Exception as e:
                # cancel all remaining tasks and wait for them to finish
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                main_logger.error(f"aborting due to error: {e}")
                if pending_inserts:
                    with Session(engine) as session:
                        for kid_i, lid_i, ans in pending_inserts:
                            insert_answer(session, kid_i, lid_i, ans, log_queue)
                        session.commit()
                    inserted += len(pending_inserts)
                    pending_inserts.clear()
                raise

            if not answer:
                continue
            kid: int = row["keyword_query_id"]
            lid: int = row["link_id"]
            main_logger.debug(f"this is the answer: {answer}")
            pending_inserts.append((kid, lid, answer))

            if len(pending_inserts) >= batch_commit:
                with Session(engine) as session:
                    for kid_i, lid_i, ans in pending_inserts:
                        insert_answer(session, kid_i, lid_i, ans, log_queue)
                    session.commit()
                inserted += len(pending_inserts)
                pending_inserts.clear()

        if pending_inserts:
            with Session(engine) as session:
                for kid_i, lid_i, ans in pending_inserts:
                    insert_answer(session, kid_i, lid_i, ans, log_queue)
                session.commit()
            inserted += len(pending_inserts)

        main_logger.info(f"\nDone. Inserted {inserted} answers.")
    finally:
        listener.stop()
        logging.shutdown()


def _evaluate_validity_worker(
    content_id_text_queue,
    result_content_id_is_valid_queue,
    stop_processing_event,
    log_queue,
):
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_evaluate_validity_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = content_id_text_queue.get(timeout=1)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return

            content_id, content_text = item
            logger.info(f"evaluate validity of {content_id}")
            clean_text = truncate_to_token_limit(
                content_text, "gpt-4o-mini", 10000
            )
            result = evaluate_validity_with_llm(clean_text, logger)
            logger.info(result)
            result_content_id_is_valid_queue.put((content_id, result))
    except Exception as e:
        print(e)
        stop_processing_event.set()
        logger.exception("something bad happened in _evaluate_validity_worker")
        raise


def _db_content_valid_writer(
    result_content_id_is_valid_queue,
    stop_processing_event,
    log_queue,
):
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_evaluate_validity_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = result_content_id_is_valid_queue.get(timeout=1)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return

            content_id, is_valid = item
            with Session(DB_ENGINE) as session:
                content = session.get(Content, content_id)
                content.is_valid = is_valid
                session.commit()

    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened in _db_content_valid_writer")


def classify_valid_content():
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/classify_valid_content.log"
    )
    content_id_text_queue = manager.Queue()
    result_content_id_is_valid_queue = manager.Queue()
    stop_processing_event = manager.Event()
    log_level = logging.DEBUG
    logger = configure_worker_logger(
        log_queue, log_level, "classify_valid_content"
    )
    set_catch_and_log_logger(logger)
    try:
        logger.info("start workers")
        n_workers = psutil.cpu_count(logical=False)
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    _evaluate_validity_worker,
                    content_id_text_queue,
                    result_content_id_is_valid_queue,
                    stop_processing_event,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            writer_future = pool.submit(
                _db_content_valid_writer,
                result_content_id_is_valid_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the contents")
            with Session(DB_ENGINE) as session:
                stmt = (
                    select(Content)
                    .where(Content.is_valid.is_(None))
                    .options(
                        load_only(Content.id),
                        defer(Content.text),
                    )
                    .execution_options(stream_results=True, yield_per=10)
                )
                for content in session.execute(stmt).scalars():
                    content_id_text_queue.put((content.id, content.text))

            for _ in range(n_workers):
                content_id_text_queue.put(None)

            logger.info("wait for _evaluate_validity_worker to finish")
            for f in worker_futures:
                f.result()
            logger.info(
                "_evaluate_validity_worker are finished, signal to stop"
            )
            result_content_id_is_valid_queue.put(None)
            stop_processing_event.set()
            logger.info("await writer task")
            writer_future.result()
            logger.info("all done with classify_valid_content url, clean up")
    finally:
        logger.info("all done with classify_valid_content, exiting")
        listener.stop()


if __name__ == "__main__":
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    classify_valid_content()
    # asyncio.run(process_unanswered_questions())
