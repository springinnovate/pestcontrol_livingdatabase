"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from typing import Any
from queue import Empty
import logging

from sqlalchemy import create_engine, select, exists
from sqlalchemy.orm import Session
import psutil

from models import (
    DB_URI,
    BaseNorm,
    Content,
    Answer,
    QuestionLink,
    Question,
    Link,
)

from dotenv import load_dotenv
from logging_tools import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)
from llm_tools import apply_llm

DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)

GLOBAL_LOG_LEVEL = logging.DEBUG


load_dotenv()  # loads the OPENAI_API_KEY


SYSTEM_PROMPT = """
You are a cautious information extraction model.

Given a QUESTION and PAGE_TEXT from a single web page (optionally SOURCE_URL), decide the answer using ONLY PAGE_TEXT. Do NOT speculate.

STRICT FORMAT
Return ONE compact JSON object with EXACTLY these top-level keys and NOTHING ELSE:
- "answer_text": string
- "reason": string (enum below)
- "evidence": array of 0–2 strings

Compliance rules:
- Keys MUST be exactly as above. NEVER use "answer", "answerKey", or any other key.
- If you cannot comply, output: {"answer_text":"unknown","reason":"page_error","evidence":[]}
- Output JSON only. No markdown, no comments, no code fences, no trailing commas.

"reason" enum:
["supported","not_enough_information","blank_page","content_unrelated","conflicting_evidence","non_english","page_error","model_uncertain"]

ALLOWED VALUES FOR "answer_text" (infer task type from QUESTION)
- Predator life stage: "larvae"|"nymphs"|"adults"|"larvae,nymphs"|"larvae,adults"|"nymphs,adults"|"larvae,nymphs,adults"|"unknown"
- Agricultural pest life stage: same set as above
- Invasiveness: "yes"|"no"|"unknown"
- Non-crop habitat use: "yes"|"no"|"unknown"
- Dietary breadth: "specialist"|"generalist"|"unknown"
- Wind dispersal: "yes"|"no"|"unknown"
- Flight ability: "yes"|"no"|"unknown"
- Seasonal migration: "yes"|"no"|"unknown"
- Pest status: "yes"|"no"|"unknown"

DECISION RULES
- Default to "answer_text":"unknown" unless PAGE_TEXT clearly supports an allowed value; then "reason":"supported".
- Exclusivity like "nymphs only" → that stage.
- Multiple supported stages → comma-join in canonical order: larvae,nymphs,adults.
- For "diet": narrow range → "specialist"; broad/diverse → "generalist"; explicit contradictions → "answer_text":"unknown","reason":"conflicting_evidence".
- "can fly" refers to adult flight unless clearly specified otherwise.
- If PAGE_TEXT is about a different entity → "content_unrelated".
- Mentions only higher/lower taxa without linkage → "not_enough_information".
- Non-English blocking judgment → "non_english".
- Blank/garbled page → "blank_page" or "page_error" as applicable.

EVIDENCE RULES
- Provide up to 2 short verbatim quotes (≤200 chars) that directly support "answer_text".
- NEVER include processing markers or placeholders (e.g., [[...]], <...>, "PDF_SKIPPED", "TRUNCATED"). If PAGE_TEXT was truncated/skipped or otherwise unusable → "evidence":[] and "reason":"page_error".
- Use [] when "answer_text" is "unknown".

NORMALIZATION
- Lowercase "answer_text".
- Use only allowed strings.
- Output exactly three keys as specified.

VALID EXAMPLE
{"answer_text":"yes","reason":"supported","evidence":["...exact quote 1...","...exact quote 2..."]}

UNKNOWN EXAMPLE
{"answer_text":"unknown","reason":"not_enough_information","evidence":[]}
""".strip()


def process_unanswered_questions() -> None:
    """Orchestrate parallel LLM answering for unanswered (question, link) pairs.

    Spawns a process pool of workers to generate answers with an LLM for each
    (question, link) that lacks an Answer, and a dedicated writer process that
    persists results into the database. Uses inter-process queues for task and
    result passing, and a shared event to coordinate shutdown.

    Steps:
      * Initialize process-safe logging and shared IPC primitives.
      * Query unanswered QuestionLink/Link pairs with non-null Content.
      * Enqueue tasks to worker processes.
      * Collect model outputs and persist them via the writer.
      * Ensure graceful shutdown on completion or error.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Propagates unexpected errors after attempting cleanup.
    """
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/iterate_content_for_llm.log", GLOBAL_LOG_LEVEL
    )
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(logger)
    question_id_link_id_queue = manager.Queue()
    result_answer_question_id_link_id_queue = manager.Queue()
    stop_processing_event = manager.Event()
    try:
        logger.info("start workers")
        n_workers = psutil.cpu_count(logical=False)
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    _answer_question_worker,
                    question_id_link_id_queue,
                    result_answer_question_id_link_id_queue,
                    stop_processing_event,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            writer_future = pool.submit(
                _insert_answer_worker,
                result_answer_question_id_link_id_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the contents")

            with Session(DB_ENGINE) as session:
                not_answered = ~exists(
                    select(Answer.id).where(
                        Answer.question_id == QuestionLink.question_id,
                        Answer.link_id == QuestionLink.link_id,
                    )
                )

                stmt = (
                    select(QuestionLink, Link)
                    .join_from(QuestionLink, Link, QuestionLink.link)
                    .where(Link.content_id.is_not(None))
                    .where(not_answered)
                    .execution_options(stream_results=True, yield_per=200)
                )

                for question_link, link in session.execute(stmt):
                    question_id_link_id_queue.put(
                        (
                            question_link.id,
                            question_link.question_id,
                            question_link.link_id,
                            link.content_id,
                        )
                    )
                    break

            for _ in range(n_workers):
                question_id_link_id_queue.put(None)

            logger.info("wait for _answer_question_worker to finish")
            for f in worker_futures:
                f.result()
            logger.info("_answer_question_worker are finished, signal to stop")
            result_answer_question_id_link_id_queue.put(None)
            stop_processing_event.set()
            logger.info("await writer task")
            writer_future.result()
            logger.info("all done with classify_valid_content url, clean up")
    finally:
        logger.info("all done with classify_valid_content, exiting")
        listener.stop()


def _answer_question_worker(
    question_id_link_id_queue: Any,
    result_answer_question_id_link_id_queue: Any,
    stop_processing_event: Any,
    log_queue: Any,
) -> None:
    """Worker process that generates an Answer for each (question, link, content).

    Consumes items of the form:
        (question_link_id: int, question_id: int, link_id: int, content_id: int)
    from the task queue, fetches the corresponding Question and Content rows,
    constructs a prompt, calls the LLM, and pushes a result tuple to the result
    queue:
        (question_link_id: int, question_id: int, link_id: int, result: dict)

    The worker exits when it receives a sentinel (None) or when the shared stop
    event is set.

    Args:
        question_id_link_id_queue: IPC queue to receive work items; items are tuples
            (question_link_id, question_id, link_id, content_id).
        result_answer_question_id_link_id_queue: IPC queue to emit LLM outputs as
            tuples (question_link_id, question_id, link_id, result_dict).
        stop_processing_event: Shared event used to signal a cooperative shutdown.
        log_queue: IPC queue for process-safe logging.

    Returns:
        None

    Raises:
        Exception: On unrecoverable errors; sets the stop event and logs the
            exception before exiting.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_answer_question_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = question_id_link_id_queue.get(timeout=1)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return
            question_link_id, question_id, link_id, content_id = item

            with Session(DB_ENGINE) as session:
                content = session.scalar(
                    select(Content).where(Content.id == content_id)
                )
                question = session.scalar(
                    select(Question).where(Question.id == question_id)
                )
                logger.debug(f"processing {question_link_id}")

                user_prompt = (
                    f"QUESTION: {question.text} PAGE_TEXT: {content.text}"
                )

                logging.getLogger("httpcore").setLevel(logging.WARNING)
                logging.getLogger("openai").setLevel(logging.WARNING)
                result = apply_llm(SYSTEM_PROMPT, user_prompt, logger, "gpt-4o")
                if result:
                    result_answer_question_id_link_id_queue.put(
                        (question_link_id, question_id, link_id, result)
                    )
    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened in _answer_question_worker")


def _insert_answer_worker(
    result_answer_question_id_link_id_queue: Any,
    stop_processing_event: Any,
    log_queue: Any,
) -> None:
    """Writer process that persists Answer rows produced by workers.

    Consumes items of the form:
        (question_link_id: int, question_id: int, link_id: int, result: dict)
    where result has keys:
        - 'answer_text': str
        - 'reason': str
        - 'evidence': list
    It inserts a new Answer row per item and commits the transaction. The writer
    exits on a sentinel (None) or when the shared stop event is set.

    Args:
        result_answer_question_id_link_id_queue: IPC queue providing result tuples
            (question_link_id, question_id, link_id, result_dict).
        stop_processing_event: Shared event used to signal a cooperative shutdown.
        log_queue: IPC queue for process-safe logging.

    Returns:
        None

    Raises:
        Exception: On database or unexpected errors; sets the stop event and logs
            the exception before exiting.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_evaluate_validity_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = result_answer_question_id_link_id_queue.get(timeout=1)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return

            question_link_id, question_id, link_id, result = item
            # result has the keys ('answer_text', 'reason', 'evidence': [])

            with Session(DB_ENGINE) as session:
                answer = Answer(
                    question_link_id=question_link_id,
                    question_id=question_id,
                    link_id=link_id,
                    answer_text=result["answer_text"],
                    reason=result["reason"],
                    evidence=result["evidence"],
                )
                session.add(answer)
                session.commit()
    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened in _insert_answer_worker")


if __name__ == "__main__":
    process_unanswered_questions()
