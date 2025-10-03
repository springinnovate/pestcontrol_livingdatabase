"""Search pipeline for unanswered questions using the DataForSEO SERP API.

This module coordinates fetching pending questions from the database,
querying the DataForSEO API for search results, and inserting new Link
and QuestionLink rows back into the database. Work is distributed across
multiple processes: one pool of workers performs keyword queries and
pushes link results onto a queue, while a dedicated worker consumes the
link results and persists them to the database.

Usage:
    Run the script directly to query all pending questions without links
    and populate new Link and QuestionLink rows. Command-line arguments
    control database URI, authentication credentials, concurrency, query
    rate limits, and test modes.
"""

import argparse
import logging
from pathlib import Path
from typing import List, NamedTuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from queue import Empty

import psutil
import httpx
import httpcore
from sqlalchemy import create_engine, select, exists, and_
from sqlalchemy.orm import Session, load_only, raiseload

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
logging.getLogger("httpcore").setLevel(logging.WARNING)


DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)


class SEOSearchResult(NamedTuple):
    """Container for a single search engine result.

    Attributes:
        url (str): Canonical URL of the search result.
        title (str): Title text of the search result.
    """

    url: str
    title: str


class KeywordPayload(NamedTuple):
    """Payload for passing a question and keyword to a SERP worker.

    Attributes:
        question_id (int): Database ID of the question being queried.
        keyword_phrase (str): Search phrase used to discover relevant links.
    """

    question_id: int
    keyword_phrase: str


class LinkResultPayload(NamedTuple):
    """Payload for passing a link result back to the database inserter.

    Attributes:
        question_id (int): Database ID of the originating question.
        url (str): URL string discovered by the SERP query.
    """

    question_id: int
    url: str


def run_serp_query(
    keyword: str, username: str, password: str, timeout: float = 30.0
) -> List[SEOSearchResult]:
    """Query the DataForSEO SERP API for organic Google results.

    Sends a POST request to the DataForSEO API with the provided keyword
    and authentication credentials. Parses the JSON response and returns
    a list of organic search results.

    Args:
        keyword (str): Search phrase to query in Google SERPs.
        username (str): DataForSEO API username.
        password (str): DataForSEO API password.
        timeout (float, optional): HTTP request timeout in seconds.
            Defaults to 30.0.

    Returns:
        List[SEOSearchResult]: A list of search result objects containing
        URL and title text for each organic result. Returns an empty list
        if no organic items are present in the response.

    Raises:
        httpx.HTTPStatusError: If the response status code indicates an error.
        httpx.RequestError: If a network or connection error occurs.
    """
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


def _safe_run_serp_query(
    keyword: str,
    username: str,
    password: str,
    retries: int = 3,
    delay: float = 2.0,
):
    """Run SERP query with retry logic for read timeouts."""
    for attempt in range(1, retries + 1):
        try:
            return run_serp_query(keyword, username, password)
        except httpcore.ReadTimeout:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)  # backoff
        except httpx.TimeoutException:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)


def serp_query_worker(
    username: str,
    password: str,
    delay: float,
    keyword_payload_queue: multiprocessing.Queue,
    link_result_payload_queue: multiprocessing.Queue,
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
        keyword_payload_queue (multiprocessing.Queue): A queue of keyword query
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
                keyword_payload = keyword_payload_queue.get(timeout=1)
                if keyword_payload is None:
                    logger.info("got a None, terminating")
                    return
                try:
                    results = _safe_run_serp_query(
                        keyword_payload.keyword_phrase,
                        username,
                        password,
                        retries=5,
                    )
                except Exception:
                    logger.warning(
                        f"could not fetch {keyword_payload.keyword_phrase}"
                    )
                    continue
                for search_result in results:
                    link_result_payload_queue.put(
                        LinkResultPayload(
                            question_id=keyword_payload.question_id,
                            url=search_result.url.strip(),
                        )
                    )
                logger.debug(
                    f"found {len(results)} urls for {keyword_payload.question_id}"
                )
            except Empty:
                logger.info("keyword queue empty; polling again")
                continue
            finally:
                if delay:
                    time.sleep(delay)
    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened, terminating")
        return


def db_link_inserter_worker(
    link_result_payload_queue: multiprocessing.Queue(),
    stop_processing_event: multiprocessing.Event(),
    log_queue: multiprocessing.Queue(),
) -> None:
    """Worker that consumes link results from a queue and inserts them into the DB.

    This function runs in its own process. It continuously pulls link
    payloads from a multiprocessing queue, ensures the corresponding
    `Link` record exists, and creates a `QuestionLink` association
    between the given `Question` and `Link`. The worker terminates when
    the stop event is set externally or when a `None` item is received
    on the queue.

    Args:
        link_result_payload_queue (multiprocessing.Queue): Queue containing
            `LinkResultPayload` objects produced by SERP workers. A `None`
            item signals termination.
        stop_processing_event (multiprocessing.Event): Event flag shared
            across workers. When set, signals all workers to stop.
        log_queue (multiprocessing.Queue): Queue used for centralized
            logging from worker processes.

    Side Effects:
        - Creates or updates `Link` rows in the database.
        - Creates new `QuestionLink` associations when needed.
        - Logs messages via the configured worker logger.
        - Terminates early and sets `stop_processing_event` if an exception
          escapes the loop.

    Raises:
        ValueError: If the referenced `Question` ID does not exist.
        RuntimeError: If a `QuestionLink` association is unexpectedly found
            when it should not be.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "db_link_inserter_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = link_result_payload_queue.get(timeout=10)
                if item is None:
                    logger.info("got none, quitting")
                    return
                logger.info(f"got this item: {item}")
                with Session(DB_ENGINE) as session, session.begin():
                    try:
                        question = session.get(Question, item.question_id)
                        if not question:
                            raise ValueError(
                                f"ID: {item.question_id} -- tried to get "
                                "question but id not found but "
                                "it got the ID from somewhere."
                            )
                        link = session.scalars(
                            select(Link).where(Link.url == item.url)
                        ).one_or_none()
                        if not link:
                            link = Link(url=item.url)
                            session.add(link)
                            session.flush()
                        ql_already_exists = session.scalar(
                            select(
                                exists().where(
                                    and_(
                                        QuestionLink.question_id == question.id,
                                        QuestionLink.link_id == link.id,
                                    )
                                )
                            )
                        )
                        if ql_already_exists:
                            raise RuntimeError(
                                f"we got to a question link but we started with "
                                f"no link: {question.id} - {link.id}"
                            )
                        ql = QuestionLink(
                            question_id=question.id, link_id=link.id
                        )
                        session.add(ql)
                        session.flush()
                        logger.info(
                            "linked question %s to link %s",
                            question.id,
                            link.id,
                        )
                    except Exception:
                        logger.exception(f"something bad happened on {item}")
                        session.rollback()
            except Empty:
                logger.info("result queue empty; polling again")
                continue

    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened")


def main():
    """Entry point for the DataForSEO search pipeline.

    Orchestrates fetching unanswered questions from the database,
    submitting keyword queries to the DataForSEO SERP API, and inserting
    resulting links back into the database. Work is distributed across
    multiple processes: several query workers fetch search results, and
    a dedicated DB worker persists the results.

    Command-line arguments:
        --db (str, optional): SQLAlchemy DB URI. Defaults to
            `models.DB_URI`.
        --auth-file (Path, optional): File containing API
            credentials as `username<newline>password`. Defaults to
            `secrets/auth`.
        --max-concurrent (int, optional): Maximum number of concurrent
            API requests. Defaults to 10.
        --qps (float, optional): Throttle for overall queries per second.
            Defaults to 20.0.
        --limit (int, optional): Limit on the number of pending questions
            to process. Defaults to None (process all).
        --keywords (str, optional): Run a one-off query for a specific
            keyword instead of processing the queue. Defaults to None.
        --question_id (int, optional): Process a single hardcoded
            question ID for testing. Defaults to None.

    Workflow:
        1. Initialize process-safe logging.
        2. Load DataForSEO credentials from the auth file.
        3. If `--keywords` is supplied, run a test query and exit.
        4. Spawn N SERP workers and one DB worker using a process pool.
        5. Fetch unanswered questions from the database and enqueue their
           keyword payloads.
        6. Wait for all workers to complete, signal the DB worker to stop,
           and block until it finishes.

    Side Effects:
        - Logs detailed progress and debug information.
        - Updates the database with new Link and QuestionLink rows.
        - May print one-off query results if `--keywords` is used.

    Raises:
        Exception: Any unhandled exception during orchestration will
        propagate after being logged.
    """
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
        logger = logging.getLogger("main")
        logger.setLevel(GLOBAL_LOG_LEVEL)
        set_catch_and_log_logger(logger)

        username, password = (
            args.auth_file.read_text(encoding="utf-8").strip().split("\n", 1)
        )

        if args.keywords:
            result = run_serp_query(args.keywords, username, password)
            print(result)
            return

        delay = 1.0 / args.qps if args.qps > 0 else 0.0

        keyword_payload_queue = manager.Queue()
        stop_processing_event = manager.Event()
        link_result_payload_queue = manager.Queue()
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
                    keyword_payload_queue,
                    link_result_payload_queue,
                    stop_processing_event,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            db_worker_future = pool.submit(
                db_link_inserter_worker,
                link_result_payload_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the questions")
            with Session(DB_ENGINE) as session:
                stmt = (
                    select(Question, Link)
                    .options(
                        load_only(Question.id),
                        load_only(Link.id, Link.url),
                        raiseload("*"),
                    )
                    .join(
                        QuestionLink,
                        QuestionLink.question_id == Question.id,
                        isouter=True,
                    )
                    .join(Link, Link.id == QuestionLink.link_id, isouter=True)
                )

                inserted = 0
                for question, link in session.execute(stmt):
                    if link:
                        logger.debug(f"link exists {link}")
                        continue
                    logger.debug(
                        f"{question.id} | {link.id}"
                        if link
                        else f"{question.id} | None"
                    )
                    keyword_payload_queue.put(
                        KeywordPayload(
                            question_id=question.id,
                            keyword_phrase=question.keyword_phrase,
                        )
                    )
                    inserted += 1
                    if args.limit is not None and inserted >= args.limit:
                        break
            for _ in range(n_workers):
                keyword_payload_queue.put(None)
            for fut in as_completed(worker_futures):
                # this explicitly raises an exception if a worker failed
                fut.result()
            logger.info("WORKERS DONE STOPPING DB LINK INSERTER")
            link_result_payload_queue.put(None)
            db_worker_future.result()
    except Exception:
        raise
    finally:
        logger.info("all done with process_unanswered_questions, exiting")
        listener.stop()


if __name__ == "__main__":
    main()
