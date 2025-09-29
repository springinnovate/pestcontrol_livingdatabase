import argparse
import logging
from pathlib import Path
from typing import List, NamedTuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from queue import Empty

import psutil
import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import (
    DB_URI,
    BaseNorm,
    Question,
    Link,
    QuestionLink,
)

from logging_tools import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)

GLOBAL_LOG_LEVEL = logging.DEBUG
logging.getLogger("httpx").setLevel(logging.WARNING)


DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)


class SEOSearchResult(NamedTuple):
    url: str
    title: str


def run_serp_query(
    keyword: str, username: str, password: str, timeout: float = 30.0
) -> List[SEOSearchResult]:
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    payload = [
        {
            "language_name": "English",
            "location_name": "United States",
            "keyword": keyword,
            "depth": 10,
        }
    ]
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, auth=(username, password), json=payload)
        r.raise_for_status()
        data = r.json()
        items = (
            data.get("tasks", [{}])[0].get("result", [{}])[0].get("items", [])
        )
        if not items:
            return []

        return [
            SEOSearchResult(url=i.get("url", ""), title=i.get("title", ""))
            for i in items
            if i.get("type") == "organic"
        ]


def get_or_create_link(session, url: str) -> int:
    link_id = session.scalar(select(Link.id).where(Link.url == url))
    if link_id:
        return link_id
    obj = Link(url=url, content_id=None)
    session.add(obj)
    session.flush()
    return obj.id


def map_link_to_question(session, question_id: int, link_id: int) -> None:
    exists = session.scalar(
        select(QuestionLink.id).where(
            QuestionLink.question_id == question_id,
            QuestionLink.link_id == link_id,
        )
    )
    if exists is None:
        session.add(QuestionLink(question_id=question_id, link_id=link_id))


def fetch_pending(
    session, limit: int | None, question_id: int | None
) -> list[tuple[int, str, str]]:
    if not question_id:
        stmt = (
            select(
                Question.id.label("question_id"),
                Question.keyword_phrase.label("keyword"),
                Question.text.label("question_text"),
            )
            .outerjoin(QuestionLink, QuestionLink.question_id == Question.id)
            .where(QuestionLink.id.is_(None))
            .order_by(Question.id.asc())
        )
    else:
        stmt = select(
            Question.id.label("question_id"),
            Question.keyword_phrase.label("keyword"),
            Question.text.label("question_text"),
        ).where(Question.id == question_id)
    if limit:
        stmt = stmt.limit(limit)
    return [(qid, kw, qtxt) for qid, kw, qtxt in session.execute(stmt).all()]


def serp_query_worker(
    username: str,
    password: str,
    delay: float,
    keyword_query_queue: multiprocessing.Queue,
    result_question_id_link_list_queue: multiprocessing.Queue,
    stop_processing_event: multiprocessing.Event,
    log_queue: multiprocessing.Queue,
):
    """Worker that consumes keyword queries from a queue and runs SERP lookups.

    This function continuously pulls keyword queries from a multiprocessing queue,
    executes search engine requests, and logs results. It stops processing either
    when the stop event is set externally or when a `None` value is encountered in
    the queue.

    The function also enforces an optional delay between requests to control query
    rate, and ensures that exceptions cause the stop event to be set so other
    workers can terminate gracefully.

    Args:
        keyword_query_queue (multiprocessing.Queue): A queue of keyword query
            strings to be consumed by this worker. A `None` item signals
            termination.
        stop_processing_event (multiprocessing.Event): An event flag shared
            between workers that signals when processing should stop.

    Side Effects:
        - Logs messages to the configured worker logger.
        - Prints SERP results to stdout.
        - May sleep between queries if a global delay is configured.
        - Sets `stop_processing_event` if an exception occurs.

    Raises:
        Exception: Any uncaught exception within the query loop is logged and
        triggers termination.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_answer_question_worker"
        )
        while not stop_processing_event.is_set():
            try:
                keyword_query = keyword_query_queue.get(timeout=1)
                if keyword_query is None:
                    logger.info("got a None, terminating")
                    return
                results = run_serp_query(keyword_query, username, password)
                raise RuntimeError(f"process {results}")
                result_question_id_link_list_queue.put(results)
                logger.debug(results)
            except Empty:
                logger.info("keyword queue empty; polling again")
                continue
            finally:
                if delay:
                    time.sleep(delay)
    except:
        stop_processing_event.set()
        logger.exception("something bad happened, terminating")
        return


def db_link_inserter_worker(
    result_question_id_link_list_queue: multiprocessing.Queue(),
    stop_processing_event: multiprocessing.Event(),
    log_queue: multiprocessing.Queue(),
) -> None:
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "db_link_inserter_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = result_question_id_link_list_queue.get(timeout=1)
                if item is None:
                    logger.info("got none, quitting")
                    return
            except Empty:
                logger.info("result queue empty; polling again")
                continue

            with Session(DB_ENGINE) as session:
                pass

    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened")


def main():
    parser = argparse.ArgumentParser(
        description="Query keywords from DB for SearchHeads with no SearchResultLink rows and populate Links & SearchResultLinks."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLAlchemy DB URI (default: models.DB_URI)",
    )
    parser.add_argument(
        "--auth-file",
        type=Path,
        default=Path("secrets/auth"),
        help="File with username\\npassword",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent API requests",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=20.0,
        help="Overall queries per second throttle",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of pending SearchHeads to process",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="To test searching one question",
    )
    parser.add_argument(
        "--question_id",
        type=int,
        default=None,
        help="Hardcode the question id for testing",
    )

    args = parser.parse_args()

    try:
        manager = multiprocessing.Manager()
        log_queue, listener = start_process_safe_logging(
            manager, "logs/dataforseo_search_pipeline.log", GLOBAL_LOG_LEVEL
        )
        logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
        set_catch_and_log_logger(logger)

        username, password = (
            args.auth_file.read_text(encoding="utf-8").strip().split("\n", 1)
        )

        if args.keywords:
            result = run_serp_query(args.keywords, username, password)
            print(result)
            return

        delay = 1.0 / args.qps if args.qps > 0 else 0.0

        keyword_query_queue = manager.Queue()
        stop_processing_event = manager.Event()
        result_question_id_link_list_queue = manager.Queue()
        logger.info("start workers")
        n_workers = psutil.cpu_count(logical=False)
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    serp_query_worker,
                    username,
                    password,
                    delay,
                    keyword_query_queue,
                    result_question_id_link_list_queue,
                    stop_processing_event,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            db_worker_future = pool.submit(
                db_link_inserter_worker,
                result_question_id_link_list_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the questions")

            for i in range(n_workers):
                logger.debug(i)
                keyword_query_queue.put(None)
            worker_status_list = [worker.result() for worker in worker_futures]
            logger.info("WORKERS DONE STOPPING DB LINK INSERTER")
            result_question_id_link_list_queue.put(None)
            answer_status = db_worker_future.result()
    except:
        raise
    finally:
        logger.info("all done with process_unanswered_questions, exiting")
        listener.stop()


if __name__ == "__main__":
    main()
