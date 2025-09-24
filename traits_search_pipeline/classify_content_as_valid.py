"""Iterate through unanswered questions to answer them."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from queue import Empty
import logging

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, defer, load_only
import psutil

from logging_tools import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)
from llm_tools import (
    truncate_to_token_limit,
    evaluate_validity_with_llm,
)
from models import (
    DB_URI,
    BaseNorm,
    Content,
)


DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)
GLOBAL_LOG_LEVEL = logging.DEBUG

load_dotenv()  # loads the OPENAI_API_KEY


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
    """Updates db's content to classify as valid/invalid."""
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
    classify_valid_content()
