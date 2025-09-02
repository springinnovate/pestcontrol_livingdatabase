"""
docker build -t apify_pcld:latest . && docker run --rm -it -v "%CD%/apify_storage":/apify_storage apify_pcld:latest

"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from multiprocessing import Manager, MPQueue
from queue import Empty
from typing import AsyncIterator, Iterable
import asyncio
import itertools
import html as html_utils
from pathlib import Path
import logging
import random
import re

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pm_extract
from playwright.async_api import (
    async_playwright,
    TimeoutError as PWTimeoutError,
    Error as PlaywrightError,
)
import psutil
from pypdf import PdfReader
from readability import Document  # readability-lxml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import trafilatura

from models import Link, DB_URI

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

PDF_MAX_PAGES = 50
BATCH_SIZE = 10


engine = create_async_engine(DB_URI, future=True, echo=False)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


def start_process_safe_logging(log_path: str, level: int = logging.INFO):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # file handler in the parent only (single writer)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s"
        )
    )

    log_queue: MPQueue = MPQueue(-1)
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root.addHandler(sh)

    return log_queue, listener


def stop_process_safe_logging(listener: logging.handlers.QueueListener):
    listener.stop()


# ------------- worker-side config -------------


def configure_worker_logger(log_queue: MPQueue, level: int, logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # remove any existing handlers to avoid duplicate logs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(level)
    logger.addHandler(qh)
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
    if "application/pdf" in (content_type or "").lower():
        return True
    u = (url or "").lower()
    return u.endswith(".pdf") or ("/pdf" in u or "download" in u) and "pdf" in u


def parse_pdf_bytes(data: bytes, max_pages: int = 20) -> str:
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


async def soft_block_detect(page) -> bool:
    try:
        title = (await page.title()) or ""
    except Exception:
        title = ""
    body = ""
    try:
        if await page.locator("body").count():
            body = await page.inner_text("body")
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


async def fetch_once(
    context, url: str, *, timeout_ms: int = 60000, block_resources: bool = True
) -> str:
    page = await context.new_page()
    try:
        if block_resources:

            async def _route(route, request):
                if request.resource_type in {"image", "font", "media"}:
                    try:
                        await route.abort()
                    except Exception:
                        await route.continue_()
                else:
                    await route.continue_()

            await page.route("**/*", _route)

        resp = await page.goto(
            url, wait_until="domcontentloaded", timeout=timeout_ms
        )
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PWTimeoutError:
            pass

        # soft block retry handled by caller (we only report state here)
        status = resp.status if resp else 0
        final_url = resp.url if resp else url
        ctype = (resp.headers.get("content-type") if resp else "") or ""

        if looks_pdf_from_headers_url(ctype, final_url):
            try:
                data = await resp.body()
            except Exception:
                data = b""
            return parse_pdf_bytes(data, max_pages=20)

        # prefer rendered HTML
        try:
            html = await page.content()
        except Exception:
            html = ""
        if not html and resp:
            try:
                html = await resp.text()
            except Exception:
                html = ""
        if not html:
            try:
                visible = await page.evaluate(
                    'document.body && document.body.innerText || ""'
                )
            except Exception:
                visible = ""
            return visible.strip()

        return extract_text(html, base_url=final_url)
    finally:
        await page.close()


async def fetch_url(
    url: str, *, timeout_ms: int = 60000, max_retries: int = 3
) -> str:
    LOGGER.info("fetch: %s", url)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = await browser.new_context(
            user_agent=UA,
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-CH-UA": '"Chromium";v="140", "Not.A/Brand";v="24", "Google Chrome";v="140"',
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": '"Windows"',
            },
        )
        try:
            for attempt in range(1, max_retries + 1):
                text = await fetch_once(
                    context, url, timeout_ms=timeout_ms, block_resources=True
                )
                if text and not await soft_block_detect(
                    await context.new_page()
                ):
                    return text
                LOGGER.info("retry %d for %s", attempt, url)
                await asyncio.sleep(0.8 + random.random() * 1.2)
            return text  # type: ignore[has-type]
        finally:
            await context.close()
            await browser.close()


async def _url_scrape_worker(
    urls_to_process_queue,
    url_to_content_queue,
    stop_processing_event,
    log_queue,
):
    browser = None
    context = None
    logger = configure_worker_logger(
        log_queue, logging.DEBUG, "_url_scrape_worker"
    )

    async def _open_browser():
        """Used to create a browser locally in case it crashes."""
        nonlocal browser, context
        if browser:
            try:
                await browser.close()
            except PlaywrightError:
                pass
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

    async with async_playwright() as pw:
        await _open_browser()
        try:
            while not stop_processing_event.is_set():
                try:
                    url = urls_to_process_queue.get(timeout=1)
                except Empty:
                    # this could be because the queue isn't filled yet or
                    # the work is done and we'll signal a flag
                    continue
                try:
                    async with context.new_page() as page:
                        response = await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=60000,
                        )
                        try:
                            await page.wait_for_load_state(
                                "networkidle", timeout=60000
                            )
                        except PWTimeoutError:
                            # page might stream, so we just wait up to
                            # 60s in case it's just gonna doomscroll
                            # we can get what's there
                            pass

                        content_type = (
                            response.headers.get("content-type")
                            if response
                            else ""
                        ) or ""
                        response_url = response.url if response else url

                        if looks_pdf_from_headers_url(
                            content_type, response_url
                        ):
                            data = await (response.body() if response else b"")
                            content = parse_pdf_bytes(
                                data, max_pages=PDF_MAX_PAGES
                            )
                        else:
                            html = await page.content()
                            content[url] = extract_text(
                                html,
                                base_url=(response_url),
                            )
                        url_to_content_queue.put((url, content))
                except PlaywrightError as e:
                    logger.warning(
                        f"Playwright error on {url}: {e}; restarting browser"
                    )
                    await _open_browser()  # re-create and continue
                except Exception as e:
                    logger.exception(f"Unhandled scrape error for {url}, {e}")
                    stop_processing_event.set()
        finally:
            try:
                if context:
                    await context.close()
                if browser:
                    await browser.close()
            except PlaywrightError:
                pass


async def scrape_urls(
    urls: Iterable[str],
    max_concurrency: int,
):

    # pass `log_queue` to each worker (e.g., via ProcessPoolExecutor args)
    manager = Manager()
    urls_to_process_queue = manager.Queue()
    url_to_content_queue = manager.Queue()
    stop_processing_event = manager.Event()
    log_queue, listener = start_process_safe_logging("logs/scraper_errors.log")
    workers = min(max_concurrency, psutil.cpu_count(logical=False))
    loop = asyncio.get_running_loop()

    try:
        writer_task = asyncio.create_task(
            db_writer(url_to_content_queue, stop_processing_event)
        )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                loop.run_in_executor(
                    pool,
                    _url_scrape_worker,
                    urls_to_process_queue,
                    url_to_content_queue,
                    log_queue,
                    stop_processing_event,
                )
                for _ in range(workers)
            ]

            await asyncio.gather(*futures)
            # all done, we tell the workers to stop
            stop_processing_event.set()
            await writer_task
    finally:
        listener.stop()


async def stream_links(max_items: int | None = None):
    async with AsyncSessionLocal() as sess:
        stmt = select(Link.id, Link.url).execution_options(stream_results=True)
        if max_items is not None:
            stmt = stmt.limit(max_items)
        result = await sess.stream(stmt)
        async for row in result:
            yield (row.id, row.url)


async def db_writer(url_to_content_queue, stop_event, logger):
    pending = []
    async with AsyncSessionLocal() as sess:
        while True:
            if (
                stop_event.is_set()
                and url_to_content_queue.empty()
                and not pending
            ):
                break
            try:
                status, url, payload = url_to_content_queue.get(timeout=0.5)
                pending.append((status, url, payload))
            except Empty:
                pass

            if pending and (len(pending) >= BATCH_SIZE or stop_event.is_set()):
                # write a batch (pseudo upsert)
                try:
                    for status, url, payload in pending:
                        logger.info(f"{status}: {url}: {payload}")
                        if status == "ok":
                            # TODO: replace with your model + upsert (example shown)
                            # sess.add(Content(url=url, text=payload))
                            pass
                        else:
                            # optionally write to an errors table
                            # sess.add(ScrapeError(url=url, message=payload))
                            pass
                    await sess.commit()
                except Exception:
                    await sess.rollback()
                    # you might want to log and requeue/skip
                finally:
                    pending.clear()


async def _main():
    scrape_urls(stream_links(1))
    # async for link in stream_links():
    #     print(link)


if __name__ == "__main__":
    asyncio.run(_main())
