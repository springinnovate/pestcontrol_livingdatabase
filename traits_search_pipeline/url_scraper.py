"""# noqa: D200, D205, D210, D415
docker build -t apify_pcld:latest . && docker run --rm -it -v "%CD%/apify_storage":/apify_storage apify_pcld:latest
"""

from __future__ import annotations
import functools
import traceback
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from multiprocessing import Queue as MPQueue, Manager
from queue import Empty
from typing import Iterable
import asyncio
import html as html_utils
from pathlib import Path
import logging
import inspect
import logging.handlers
import re

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pm_extract
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeoutError,
    Error as PlaywrightError,
)
import psutil
from pypdf import PdfReader
from readability import Document  # readability-lxml
from sqlalchemy import select, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
import trafilatura

from models import Link, DB_URI

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

PDF_MAX_PAGES = 50
BATCH_SIZE = 10


engine = create_async_engine(
    DB_URI,  # must be sqlite+aiosqlite:///path/to.db
    echo=False,
    pool_pre_ping=True,
    connect_args={"timeout": 30},  # seconds to wait on SQLite file locks
)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

sync_engine = create_engine(
    DB_URI.replace("+aiosqlite", ""),  # to make sure it is sync
    echo=False,
    pool_pre_ping=True,
    connect_args={"timeout": 30},
)
SessionLocal = sessionmaker(
    bind=sync_engine, expire_on_commit=False, class_=Session
)

_DEFAULT_CATCH_LOGGER: logging.Logger | None = None


def set_catch_and_log_logger(logger: logging.Logger) -> None:
    """Helpful to set the 'global' logger depending on what process calls it."""
    global _DEFAULT_CATCH_LOGGER
    _DEFAULT_CATCH_LOGGER = logger


def _resolve_logger(logger: logging.Logger | str | None) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    if isinstance(logger, str):
        return logging.getLogger(logger)
    if _DEFAULT_CATCH_LOGGER is not None:
        return _DEFAULT_CATCH_LOGGER
    return logging.getLogger(__name__)


def catch_and_log(
    logger: logging.Logger | str | None = None, *, level: int = logging.ERROR
):
    """Wrapper to log any exception that's raised, helpful in subprocesses."""
    log = _resolve_logger(logger)

    def _fmt(func_name: str, e: BaseException) -> str:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return f"Unhandled exception in {func_name}:\n{tb}"

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log.log(level, _fmt(func.__qualname__, e))
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log.log(level, _fmt(func.__qualname__, e))
                    raise

            return sync_wrapper

    return decorator


async def _init_sqlite():
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA busy_timeout=30000")


def start_process_safe_logging(log_path: str, level: int = logging.INFO):
    """Set up logging that works across processes with a queue."""
    format_str = (
        "%(asctime)s %(processName)s %(levelname)s "
        "[%(filename)s:%(lineno)d] %(name)s: %(message)s"
    )

    class _FlushRotatingFileHandler(logging.handlers.RotatingFileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # file handler in the parent only (single writer)
    file_handler = _FlushRotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(format_str))

    manager = Manager()
    log_queue = manager.Queue()
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(format_str))
        root.addHandler(sh)

    return log_queue, listener, manager


def configure_worker_logger(
    log_queue: MPQueue, level: int, logger_name: str
) -> logging.Logger:
    """Used per process to get the local logger that's connected globally."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(level)
    logger.addHandler(qh)
    set_catch_and_log_logger(logger)
    return logger


def extract_text(html: str, base_url: str | None = None) -> str:
    """Extract readable text content from raw HTML, using multiple fallbacks.

    This function attempts to extract meaningful text from a web page's HTML.
    It first tries `trafilatura.extract` for high-quality article extraction,
    then falls back to `readability.Document`, then BeautifulSoup stripping,
    and finally a crude regex-based tag removal.

    Args:
        html (str): The raw HTML source of a web page. This should be a full
            HTML document or fragment that you want to extract visible text
            from.
        base_url (str | None): The canonical or final URL of the page.
            This is passed to `trafilatura.extract` to improve extraction
            quality (e.g., resolving relative links, using URL heuristics).
            If not provided, extraction is attempted without URL context.

    Returns:
        str: Cleaned and extracted plain text content from the HTML. If no
        readable content can be extracted, returns a placeholder string or
        an empty string.

    Notes:
        - Script, style, and noscript tags are always removed.
        - Trafilatura may use `base_url` for URL resolution and better
          boilerplate removal.
        - Multiple extraction methods are attempted in order of reliability.
    """
    try:
        txt = trafilatura.extract(
            html, url=base_url, include_comments=False, include_tables=False
        )
        if txt and txt.strip():
            return txt.strip()
    except Exception:
        pass
    try:
        summary_html = Document(html).summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        pass
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        pass
    # ultra-fallback: strip tags crudely

    scrubbed_html = re.sub(
        r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html
    )
    scrubbed_html = re.sub(
        r"(?i)</?(p|div|br|li|h[1-6]|tr|th|td)\b[^>]*>", "\n", scrubbed_html
    )
    text = re.sub(r"(?s)<[^>]+>", " ", scrubbed_html)
    text = html_utils.unescape(text)
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\f\v]+", " ", text))
    return text.strip()


def looks_pdf_from_headers_url(content_type: str, url: str) -> bool:
    """Helper to see if content or url looks like a pdf."""
    if "application/pdf" in (content_type or "").lower():
        return True
    u = (url or "").lower()
    return u.endswith(".pdf") or ("/pdf" in u or "download" in u) and "pdf" in u


def parse_pdf_bytes(data: bytes, max_pages: int = 20) -> str:
    """Helper to strip out the pdf from raw bytes."""
    try:
        reader = PdfReader(BytesIO(data))
        if len(reader.pages) > max_pages:
            return f"[[PDF_SKIPPED_TOO_LONG:{len(reader.pages)}]]"
        parts: list[str] = []
        for i in range(len(reader.pages)):
            t = (reader.pages[i].extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts) if parts else "[[PDF_EMPTY_TEXT]]"
    except Exception:
        try:
            return pm_extract(BytesIO(data), maxpages=max_pages) or ""
        except Exception:
            return "[[PDF_PARSE_ERROR]]"


def soft_block_detect(page) -> bool:
    """Helper to try to determine if a page is soft blocked."""
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    body = ""
    try:
        if page.locator("body").count():
            body = page.inner_text("body")
    except Exception:
        pass
    blob = (title + "\n" + body).lower()
    return any(
        s in blob
        for s in (
            "temporarily unavailable",
            "just a moment",
            "verify you are a human",
            "are you a human",
            "access denied",
            "rate limit",
            "unusual traffic",
        )
    )


def _url_scrape_worker(
    url_link_tuples_process_queue,
    url_to_content_queue,
    stop_processing_event,
    log_queue,
):
    """Processes given urls to scrape respective webpages."""
    logger = configure_worker_logger(
        log_queue, logging.DEBUG, "_url_scrape_worker"
    )

    browser = None
    context = None
    pages_done = 0

    def _open_browser():
        nonlocal browser, context
        if browser:
            try:
                logger.debug("closing existing browser before reopen")
                browser.close()
            except PlaywrightError as e:
                logger.debug(f"ignoring browser close error during reopen: {e}")
        logger.info("launching browser")
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        logger.debug("creating new context")
        context = browser.new_context(
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        logger.info("browser/context ready")

    logger.info("worker start")
    with sync_playwright() as pw:
        _open_browser()
        try:
            while not stop_processing_event.is_set():
                try:
                    payload = url_link_tuples_process_queue.get(timeout=1)
                    if payload is None:
                        # sentinel to indicate no more data
                        break
                    link_id, url = payload
                except Empty:
                    logger.debug("task queue empty; polling again")
                    continue

                logger.info(f"processing url: {url}")
                try:
                    with context.new_page() as page:
                        logger.debug(f"goto domcontentloaded: {url}")
                        response = page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=60000,
                        )
                        try:
                            logger.debug("waiting for networkidle")
                            page.wait_for_load_state(
                                "networkidle", timeout=60000
                            )
                        except PWTimeoutError:
                            logger.debug(
                                "networkidle timeout; proceeding with current DOM"
                            )

                        content_type = (
                            response.headers.get("content-type")
                            if response
                            else ""
                        ) or ""
                        response_url = response.url if response else url
                        logger.debug(
                            f"content-type={content_type!r} final_url={response_url}"
                        )

                        if looks_pdf_from_headers_url(
                            content_type, response_url
                        ):
                            logger.debug("detected pdf; fetching body")
                            data = response.body() if response else b""
                            content = parse_pdf_bytes(
                                data, max_pages=PDF_MAX_PAGES
                            )
                            logger.debug(
                                f'pdf parsed ({len(content) if isinstance(content, str) else "n/a"} chars)'
                            )
                        else:
                            logger.debug("extracting html content")
                            html = page.content()
                            content = extract_text(html, base_url=response_url)
                            logger.debug(
                                f"html extracted ({len(content)} chars)"
                            )

                        url_to_content_queue.put(("ok", link_id, url, content))
                        pages_done += 1
                        logger.info(
                            f"queued result for {url} (total processed: {pages_done})"
                        )

                except PlaywrightError as e:
                    logger.warning(
                        f"playwright error on {url}: {e}; restarting browser/context"
                    )
                    _open_browser()
                except Exception as e:
                    logger.exception(f"unhandled scrape error for {url}: {e}")
                    stop_processing_event.set()
                    # do not break immediately; let finally close resources

        finally:
            logger.info("worker shutting down; closing context/browser")
            try:
                if context:
                    context.close()
            except PlaywrightError as e:
                logger.debug(f"ignoring context close error: {e}")
            try:
                if browser:
                    browser.close()
            except PlaywrightError as e:
                logger.debug(f"ignoring browser close error: {e}")
            logger.info(f"worker exit after processing {pages_done} pages")


@catch_and_log()
def scrape_urls(
    url_generator: Iterable[str],
    max_concurrency: int,
    manager: Manager,
    log_queue: MPQueue,
    listener: logging.handlers.QueueListener,
):
    """Driver to launch workers to scrape the urls in the generator."""
    url_link_tuples_process_queue = manager.Queue()
    url_to_content_queue = manager.Queue()
    stop_processing_event = manager.Event()
    logger = configure_worker_logger(log_queue, logging.DEBUG, "scrape_urls")
    n_workers = min(max_concurrency, psutil.cpu_count(logical=False))
    try:
        logger.info("start writer")
        logger.info("start _url_scrape_workers")
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    _url_scrape_worker,
                    url_link_tuples_process_queue,
                    url_to_content_queue,
                    stop_processing_event,
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
                logger.info(f'queuing "{url_link_tuple}"')
                url_link_tuples_process_queue.put(url_link_tuple)

            for _ in range(n_workers):
                url_link_tuples_process_queue.put(None)

            logger.info("wait for _url_scrape_workers to finish")
            for f in worker_futures:
                f.result()
            logger.info("_url_scrape_workers are finished, signal to stop")
            stop_processing_event.set()
            logger.info("await writer task")
            writer_future.result()
            logger.info("all done with scrape url, clean up")
    finally:
        listener.stop()
        logger.info("all done with scrape url, exiting")


def stream_links(log_queue: MPQueue, max_items: int | None = None):
    """Generator for links that need fetching in the DB."""
    logger = configure_worker_logger(log_queue, logging.DEBUG, "stream_links")
    logger.info("open AsyncSession")
    with SessionLocal() as sess:
        logger.info("build statement")
        stmt = select(Link.id, Link.url).execution_options(
            stream_results=True,
            yield_per=500,
        )
        if max_items is not None:
            logger.info(f"limit {max_items}")
            stmt = stmt.limit(max_items)

        logger.info("create stream (may wait if DB is locked)")
        result = sess.stream(stmt)

        yielded = 0
        logger.info("iterate remaining rows...")
        for row in result:
            yielded += 1
            if yielded % 100 == 0:
                logger.info(f"yielded {yielded} rows so far...")
            logger.info(f"yielding row {yielded}: id={row.id}, url={row.url}")
            yield (row.id, row.url)

        logger.info(f"done, total yielded={yielded}")


def db_writer(url_to_content_queue, stop_event, log_queue):
    """Watch the queue to see links/content come back ready to insert in the db."""
    pending = []
    logger = configure_worker_logger(log_queue, logging.DEBUG, "db_writer")
    logger.info("starting up")
    with SessionLocal() as sess:
        while True:
            if (
                stop_event.is_set()
                and url_to_content_queue.empty()  # noqa: W503
                and not pending  # noqa: W503
            ):
                logger.info(
                    "stop is set, content is empty and nothing pending, quitting"
                )
                break
            try:
                status, link_id, url, payload = url_to_content_queue.get(
                    timeout=0.5
                )
                pending.append((status, url, payload))
            except Empty:
                logger.info("no content to write, waiting for some...")

            if pending and (len(pending) >= BATCH_SIZE or stop_event.is_set()):
                # write a batch (pseudo upsert)
                try:
                    for status, url, payload in pending:
                        logger.info(f"{status}: {url}: {payload}")
                        if status == "ok":
                            logger.error("ok status not implemented")
                            # TODO: replace with your model + upsert (example shown)
                            # sess.add(Content(url=url, text=payload))
                        else:
                            # optionally write to an errors table
                            # sess.add(ScrapeError(url=url, message=payload))
                            logger.error("error status not implemented")
                    sess.commit()
                except Exception:
                    sess.rollback()
                finally:
                    pending.clear()


async def _main():
    log_queue, listener, manager = start_process_safe_logging(
        "logs/scraper_errors.log"
    )
    main_logger = configure_worker_logger(log_queue, logging.INFO, "main")
    set_catch_and_log_logger(main_logger)
    await _init_sqlite()
    scrape_urls(
        stream_links(log_queue, 1),
        1,
        manager,
        log_queue,
        listener,
    )


if __name__ == "__main__":
    asyncio.run(_main())
