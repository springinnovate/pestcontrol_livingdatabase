import logging
from queue import Empty
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from dotenv import load_dotenv

from sqlalchemy import create_engine, select, or_, exists, event, text, and_
from sqlalchemy.orm import Session, aliased, defer

from models import Question, Link, Content, QuestionLink, Answer, DB_URI
from llm_tools import evaluate_validity_with_llm

from logging_tools import (
    configure_worker_logger,
    catch_and_log,
    set_catch_and_log_logger,
    start_process_safe_logging,
)

load_dotenv()

DB_ENGINE = create_engine(DB_URI)


def _evaluate_validity_worker(
    content_id_text_queue,
    result_is_valid__queue,
    stop_processing_event,
    log_queue,
):
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_evaluate_validity_worker"
    )
    while not stop_processing_event.is_set():
        try:
            item = content_id_text_queue.get(timeout=1)
        except Empty:
            logger.info("task queue empty; polling again")
            continue

        content_id, content_text = item
        logger.info(f"evaluate validity of {content_id}")
        result = evaluate_validity_with_llm(content_text, logger)
        logger.info(result)
        break


def main():
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/content_valid.log"
    )
    log_level = "DEBUG"
    logger = configure_worker_logger(log_queue, log_level, "main")
    set_catch_and_log_logger(logger)
    try:
        logger.info("start scrape_workers")
        n_workers = 1
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    _url_scrape_worker,
                    url_link_tuples_process_queue,
                    url_to_content_queue,
                    stop_processing_event,
                    flaresolver_semaphore,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            writer_future = pool.submit(
                db_writer,
                url_to_content_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the urls")
            for url_link_tuple in url_generator:
                url_link_tuples_process_queue.put(url_link_tuple)

            for _ in range(n_workers):
                url_link_tuples_process_queue.put(None)

            logger.info("wait for _url_scrape_workers to finish")
            for f in worker_futures:
                f.result()
            logger.info("_url_scrape_workers are finished, signal to stop")
            url_to_content_queue.put(None)
            stop_processing_event.set()
            logger.info("await writer task")
            writer_future.result()
            logger.info("all done with scrape url, clean up")
    finally:
        listener.stop()
        logger.info("all done with scrape url, exiting")

    result = evaluate_validity_with_llm(test_text, logger)
    logger.info(result)

    # with Session(DB_ENGINE) as session:
    #     stmt = (
    #         select(Content)
    #         .where(Content.is_valid.is_(None))
    #         .execution_options(stream_results=True, yield_per=10)
    #     )
    #     for content in session.execute(stmt).scalars():
    #         logger.info(content.text)
    #         result = evaluate_validity_with_llm(content.text, logger)
    #         logger.info(result)
    #         break

    listener.stop()


if __name__ == "__main__":
    main()
