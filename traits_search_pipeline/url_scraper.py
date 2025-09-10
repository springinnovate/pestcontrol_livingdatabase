from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from multiprocessing import Queue as MPQueue, Manager
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from typing import Iterable, Tuple, Iterator
import asyncio
import functools
import hashlib
import html as html_utils
import inspect
import logging
import logging.handlers
import os
import re
import traceback

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
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import trafilatura

from models import Content, Link, DB_URI


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

PDF_MAX_PAGES = 50
BATCH_SIZE = 10
NAV_TIMEOUT_MS = 60000
DOWNLOAD_SENTINEL = "Download is starting"

sync_engine = create_engine(
    DB_URI,
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
    logger: logging.Logger | str | None = None, *, level: int = logging.DEBUG
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


@catch_and_log()
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


@catch_and_log()
def _url_scrape_worker(
    url_link_tuples_process_queue,
    url_to_content_queue,
    stop_processing_event,
    log_queue,
):
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )

    browser = None
    context = None
    processed_count = 0

    def _open_browser():
        nonlocal browser, context
        # close stale handles
        for closer, label in (
            (getattr(context, "close", None), "context"),
            (getattr(browser, "close", None), "browser"),
        ):
            try:
                closer and closer()
            except PlaywrightError as e:
                logger.debug(f"ignoring {label} close error during reopen: {e}")

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
            accept_downloads=True,  # critical for handling attachment navigations
        )
        logger.info("browser/context ready")

    def _read_download_bytes(download) -> bytes:
        # prefer the already-downloaded path; otherwise stream to a temp file
        try:
            path = download.path()
        except PlaywrightError:
            path = None
        if path:
            return Path(path).read_bytes()
        with NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name
        try:
            download.save_as(tmp_name)
            return Path(tmp_name).read_bytes()
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def _is_probable_pdf_url(url: str) -> bool:
        lo = url.lower()
        return (
            lo.endswith(".pdf") or "contenttype=pdf" in lo or "format=pdf" in lo
        )

    def _is_pdf_download(download, src_url: str) -> bool:
        filename = (getattr(download, "suggested_filename", "") or "").lower()
        if filename.endswith(".pdf"):
            return True
        return looks_pdf_from_headers_url("", src_url)

    def _download_via_goto(page, target_url: str) -> Tuple[str, str]:
        # Expect a download and swallow the Page.goto exception inside the 'with' block.
        with page.expect_download(timeout=NAV_TIMEOUT_MS) as download_info:
            try:
                page.goto(
                    target_url, wait_until="commit", timeout=NAV_TIMEOUT_MS
                )
            except PlaywrightError as e:
                if DOWNLOAD_SENTINEL not in str(e):
                    raise
                # swallow the expected 'Download is starting' so that expect_download can resolve
        download = download_info.value
        src_url = getattr(download, "url", None) or target_url
        blob = _read_download_bytes(download)
        if _is_pdf_download(download, src_url):
            text = parse_pdf_bytes(blob, max_pages=PDF_MAX_PAGES)
        else:
            try:
                text = blob.decode("utf-8", "ignore")
            except Exception:
                text = ""
        return text, src_url

    def _fetch_bytes_via_request(target_url: str) -> Tuple[str, bytes]:
        # Fallback path that avoids renderer entirely
        resp = context.request.get(target_url, timeout=NAV_TIMEOUT_MS)
        if not resp.ok:
            raise RuntimeError(
                f"GET {target_url} failed: {resp.status} {resp.status_text()}"
            )
        return resp.headers.get("content-type", ""), resp.body()

    def _extract_text_from_page(page, response, url: str) -> Tuple[str, str]:
        try:
            page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT_MS)
        except PWTimeoutError:
            logger.debug("networkidle timeout; proceeding with current DOM")

        content_type = (
            response.headers.get("content-type") if response else ""
        ) or ""
        final_url = response.url if response else url
        logger.debug(f"content-type={content_type!r} final_url={final_url}")

        if looks_pdf_from_headers_url(content_type, final_url):
            logger.debug("detected pdf; fetching body via response")
            data = response.body() if response else b""
            text_content = parse_pdf_bytes(data, max_pages=PDF_MAX_PAGES)
        else:
            logger.debug("extracting html content")
            html = page.content()
            text_content = extract_text(html, base_url=final_url)

        return text_content, final_url

    def _navigate_and_collect_text(target_url: str) -> Tuple[str, str]:
        # Prefer direct download handling when the URL clearly points to a file.
        page = context.new_page()
        try:
            if _is_probable_pdf_url(target_url):
                logger.debug("url looks like a file; handling as download")
                return _download_via_goto(page, target_url)

            logger.debug(f"goto domcontentloaded: {target_url}")
            response = page.goto(
                target_url,
                wait_until="domcontentloaded",
                timeout=NAV_TIMEOUT_MS,
            )
            return _extract_text_from_page(page, response, target_url)

        except PlaywrightError as e:
            msg = str(e)
            if DOWNLOAD_SENTINEL in msg:
                logger.debug(
                    "download detected from goto exception; handling as download"
                )
                try:
                    return _download_via_goto(page, target_url)
                except PlaywrightError as e2:
                    # If even the download path fails, try request client as a final fallback.
                    logger.debug(
                        f"download via goto failed: {e2}; falling back to request client"
                    )
                    ctype, blob = _fetch_bytes_via_request(target_url)
                    if looks_pdf_from_headers_url(ctype or "", target_url):
                        return (
                            parse_pdf_bytes(blob, max_pages=PDF_MAX_PAGES),
                            target_url,
                        )
                    try:
                        return blob.decode("utf-8", "ignore"), target_url
                    except Exception:
                        return "", target_url
            # If it's a different Playwright error, rethrow to outer handler.
            raise
        finally:
            try:
                page.close()
            except PlaywrightError:
                pass

    logger.info("worker start")
    with sync_playwright() as pw:
        _open_browser()
        try:
            while not stop_processing_event.is_set():
                try:
                    item = url_link_tuples_process_queue.get(timeout=1)
                except Empty:
                    logger.info("task queue empty; polling again")
                    continue

                if item is None:
                    logger.info("received sentinel; exiting")
                    break

                link_id, target_url = item
                logger.info(f"processing url: {target_url}")

                try:
                    text_content, final_url = _navigate_and_collect_text(
                        target_url
                    )
                    logger.info(
                        f"this is the final url {final_url} and text content {text_content}"
                    )
                    url_to_content_queue.put(
                        (link_id, target_url, text_content)
                    )
                    processed_count += 1
                    logger.info(
                        f"queued result for {target_url} (total processed: {processed_count})"
                    )
                except PlaywrightError as e:
                    logger.warning(
                        f"playwright error on {target_url}: {e}; restarting browser/context\n\n"
                        + "".join(
                            traceback.format_exception(
                                type(e), e, e.__traceback__
                            )
                        )
                    )
                    _open_browser()
                except Exception as e:
                    logger.exception(
                        f"unhandled scrape error for {target_url}: {e}"
                    )
                    stop_processing_event.set()
        except Exception:
            logger.exception("unexpected failure in worker loop")
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
            logger.info(f"worker exit after processing {processed_count} pages")


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
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "scrape_urls")
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
            url_to_content_queue.put(None)
            stop_processing_event.set()
            logger.info("await writer task")
            writer_future.result()
            logger.info("all done with scrape url, clean up")
    finally:
        listener.stop()
        logger.info("all done with scrape url, exiting")


@catch_and_log()
def stream_links(
    log_queue: MPQueue, max_items: int = None
) -> Iterator[Tuple[int, str, str | None]]:
    def _is_error_like(text: str | None) -> bool:
        if text is None:
            return True
        t = text.strip()
        if not t:
            return True
        if t.startswith("Error:"):
            return True
        if "[[" in t:
            return True
        if "Traceback" in t:
            return True
        if "HTTP 4" in t or "HTTP 5" in t:
            return True
        return False

    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "stream_links"
    )
    logger.info("open Session")
    with SessionLocal() as sess:  # type: Session
        logger.info("build statement (links + content)")
        stmt = (
            select(Link.id, Link.url, Content.text)
            .outerjoin(Content, Link.content_id == Content.id)
            .execution_options(stream_results=True, yield_per=500)
        )

        logger.info("create stream (may wait if DB is locked)")
        result = sess.execute(stmt)

        yielded = 0
        scanned = 0
        logger.info("iterate rows...")
        for row in result:
            scanned += 1
            link_id, url, text = row
            if _is_error_like(text):
                yielded += 1
                if yielded % 100 == 0:
                    logger.info(
                        f"yielded {yielded} error-like rows so far (scanned={scanned})..."
                    )
                logger.debug(
                    f"yielding id={link_id}, url={url}, content_is_none={text is None}"
                )
                yield (link_id, url)
                if yielded == max_items:
                    logger.info(
                        f"terminating early after {yielded} items of {max_items}"
                    )
                    break
        logger.info(f"done, scanned={scanned}, yielded={yielded}")
    logger.info("terminated")


@catch_and_log()
def db_writer(url_to_content_queue, stop_event, log_queue):
    """Watch the queue to see links/content come back ready to insert in the db."""
    pending = []
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "db_writer")
    logger.info("starting up")
    while not stop_event.is_set():
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
            payload = url_to_content_queue.get()
            if payload is None:
                logger.info("got a sentinel, all work done, shutting down...")
                stop_event.set()
            else:
                link_id, url, content = payload
                pending.append((link_id, url, content))
        except Empty:
            logger.info("no content to write, waiting for some...")

        if pending and (len(pending) >= BATCH_SIZE or stop_event.is_set()):
            logger.info(f"attempting to write {len(pending)} content items")
            with SessionLocal() as sess:
                # write a batch (pseudo upsert)
                try:
                    for link_id, url, content in pending:
                        content_hash = hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest()

                        # insert the new content if it does not exist
                        sess.execute(
                            sqlite_insert(Content)
                            .values(content_hash=content_hash, text=content)
                            .on_conflict_do_nothing(
                                index_elements=["content_hash"]
                            )
                        )

                        content_id = sess.execute(
                            select(Content.id).where(
                                Content.content_hash == content_hash
                            )
                        ).scalar_one()

                        # insert the link or update its content id if it
                        # already exists
                        sess.execute(
                            sqlite_insert(Link)
                            .values(url=url, content_id=content_id)
                            .on_conflict_do_update(
                                index_elements=["url"],
                                set_={"content_id": content_id},
                            )
                        )
                        logger.info(f"{link_id}: {url}: {content}")
                    sess.commit()
                except Exception:
                    logger.exception("something bad happened on insert")
                    sess.rollback()
                finally:
                    pending.clear()
    logger.info("terminated")


async def _main():
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    log_queue, listener, manager = start_process_safe_logging(
        "logs/scraper_errors.log"
    )
    main_logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(main_logger)
    scrape_urls(
        stream_links(log_queue),
        100,
        manager,
        log_queue,
        listener,
    )


if __name__ == "__main__":
    asyncio.run(_main())
