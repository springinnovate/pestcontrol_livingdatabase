"""Make sure flaresolver is running first.

docker run -d --name=flaresolverr -p 8191:8191 -e LOG_LEVEL=info --restart unless-stopped ghcr.io/flaresolverr/flaresolverr:latest

"""

from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from multiprocessing import Queue as MPQueue, Manager
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from typing import Iterable, Tuple, Iterator, Optional
import asyncio
import hashlib
import html as html_utils
import logging
import logging.handlers
import os
import re
import traceback
import requests
from urllib.parse import urlparse
from contextlib import contextmanager, ExitStack
import threading

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pm_extract
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeoutError,
    Error as PlaywrightError,
)
import psutil
from dotenv import load_dotenv
from pypdf import PdfReader
from readability import Document  # readability-lxml
from sqlalchemy import select, create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import trafilatura

from models import Content, Link, DB_URI
from logging_tools import (
    configure_worker_logger,
    catch_and_log,
    set_catch_and_log_logger,
    start_process_safe_logging,
)
from llm_tools import guess_if_error_text

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

PDF_MAX_PAGES = 50
BATCH_SIZE = 10
NAV_TIMEOUT_MS = 180000
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

FLARESOLVERR_ENDPOINT = os.getenv("FLARESOLVERR_ENDPOINT")
FLARESOLVERR_SESSION_TTL_MIN = int(
    os.getenv("FLARESOLVERR_SESSION_TTL_MIN", "30")
)

# Optional: comma-separated domains to always route via FS (suffix match), e.g. 'cloudflaredsite.com, ddos-guard.example'
FLARESOLVERR_ALWAYS_HOSTS = tuple(
    h.strip().lower()
    for h in os.getenv("FLARESOLVERR_ALWAYS_HOSTS", "").split(",")
    if h.strip()
)


class FlareSolverrClient:
    def __init__(
        self,
        *,
        shared_semaphore=None,  # multiprocessing.Manager().Semaphore proxy
        local_max_concurrency: int = 1,  # per-process/thread concurrency
        request_timeout_sec: float = 120.0,
    ):
        try:
            self._http = None
            self.endpoint = FLARESOLVERR_ENDPOINT
            self._sessions: dict[str, str] = {}
            self._proc_sem = shared_semaphore
            self._thread_sem = threading.BoundedSemaphore(
                max(1, int(local_max_concurrency))
            )
            self._request_timeout_sec = float(request_timeout_sec)
        except Exception as e:
            print("FlareSolverrClient.__init__ ERROR:")
            print("  exception=", repr(e))
            print("  traceback:\n" + "".join(traceback.format_exc()))
            raise

    def _session(self):
        if self._http is None:
            self._http = requests.Session()
        return self._http

    def _post(self, payload: dict) -> dict:
        if not self.endpoint:
            raise RuntimeError("FLARESOLVERR_ENDPOINT not configured")
        r = self._session().post(
            self.endpoint, json=payload, timeout=self._request_timeout_sec
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f'FlareSolverr error: {data.get("message")}')
        return data

    def ensure_session(
        self, key: str, upstream_proxy: Optional[str] = None
    ) -> str:
        sid = self._sessions.get(key)
        if sid:
            return sid
        payload: dict = {"cmd": "sessions.create"}
        if upstream_proxy:
            pu = urlparse(upstream_proxy)
            if pu.username or pu.password:
                base = f"{pu.scheme}://{pu.hostname}"
                if pu.port:
                    base += f":{pu.port}"
                proxy_obj: dict[str, str] = {"url": base}
                if pu.username:
                    proxy_obj["username"] = pu.username
                if pu.password:
                    proxy_obj["password"] = pu.password
            else:
                proxy_obj = {"url": upstream_proxy}
            payload["proxy"] = proxy_obj
        data = self._post(payload)
        sid = data.get("session")
        if not sid:
            raise RuntimeError("FlareSolverr did not return session id")
        self._sessions[key] = sid
        return sid

    @contextmanager
    def _acquire(self, sem):
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def request_get(
        self,
        url: str,
        session_key: Optional[str] = None,
        upstream_proxy: Optional[str] = None,
        max_timeout_ms: int = 60000,
        session_ttl_minutes: Optional[int] = None,
    ) -> dict:
        succeeded = False
        try:
            payload: dict = {
                "cmd": "request.get",
                "url": url,
                "maxTimeout": max_timeout_ms,
            }
            if session_key:
                payload["session"] = self.ensure_session(
                    session_key, upstream_proxy=upstream_proxy
                )
                if session_ttl_minutes:
                    payload["session_ttl_minutes"] = session_ttl_minutes
            elif upstream_proxy:
                payload["proxy"] = {"url": upstream_proxy}

            with self._proc_sem:
                result = self._post(payload)["solution"]
                succeeded = True
                return result

        except Exception as e:
            print("FlareSolverrClient.request_get ERROR:")
            print(
                f"  url={url!r} session_key={session_key!r} upstream_proxy_set={bool(upstream_proxy)}"
            )
            print("  exception=", repr(e))
            print("  traceback:\n" + "".join(traceback.format_exc()))
            raise
        finally:
            if succeeded:
                print("FlareSolverrClient.request_get SUCCESS:")
                print(f"  url={url!r} session_key={session_key!r}")

    def destroy_all(self) -> None:
        for sid in list(self._sessions.values()):
            try:
                self._post({"cmd": "sessions.destroy", "session": sid})
            except Exception as e:
                print("FlareSolverrClient.destroy_all ERROR:", repr(e))
        self._sessions.clear()


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
    flaresolver_semaphore,
    log_queue,
):
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )

    browser = None
    context = None
    processed_count = 0

    logger.info("flare solver launch attempt")
    fs_client = FlareSolverrClient(
        local_max_concurrency=1,
        shared_semaphore=flaresolver_semaphore,
    )
    logger.info("flare solver launched")

    print("before if")
    sys.stdout.flush()
    if fs_client is None:
        print("fs_client is None")
        sys.stdout.flush()
    print("after if, before logger.info")
    sys.stdout.flush()
    logger.info("boot reached")  # if it hangs here, it’s logging

    if fs_client is None:
        logger.info("fs client is none")
        logger.error("flaresolver did not load")
        raise RuntimeError("flaresolver did not load")
    logger.info("fs client is good")

    def _cookies_to_playwright(
        fs_cookies: list[dict], url_for_default: str
    ) -> list[dict]:
        host = urlparse(url_for_default).hostname or ""
        out: list[dict] = []
        for c in fs_cookies or []:
            item = {
                "name": c.get("name", ""),
                "value": c.get("value", ""),
                "domain": c.get("domain") or host,
                "path": c.get("path") or "/",
                "httpOnly": bool(c.get("httpOnly", False)),
                "secure": bool(c.get("secure", False)),
            }
            if c.get("expires"):
                try:
                    item["expires"] = int(c["expires"])
                except Exception:
                    pass
            if c.get("sameSite"):
                item["sameSite"] = c["sameSite"]
            out.append(item)
        return out

    def _should_use_flaresolverr(
        target_url: str, page=None, response=None
    ) -> bool:
        if not fs_client:
            return False
        host = (urlparse(target_url).hostname or "").lower()
        if FLARESOLVERR_ALWAYS_HOSTS and any(
            host.endswith(suf) for suf in FLARESOLVERR_ALWAYS_HOSTS
        ):
            return True
        try:
            if page is not None and soft_block_detect(page):
                return True
        except Exception:
            pass
        try:
            if response is not None:
                st = getattr(response, "status", None)
                if isinstance(st, int) and st in (403, 429):
                    return True
        except Exception:
            pass
        return False

    def _ctx_from_fs_solution(sol: dict, base_url: str):
        ua = sol.get("userAgent") or USER_AGENT
        ctx = browser.new_context(
            user_agent=ua,
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            accept_downloads=True,
        )
        if sol.get("cookies"):
            ctx.add_cookies(_cookies_to_playwright(sol["cookies"], base_url))
        return ctx

    def _fs_fetch(
        target_url: str, *, force_download: bool = False
    ) -> tuple[str, str]:
        sol = fs_client.request_get(
            target_url,
            max_timeout_ms=NAV_TIMEOUT_MS,
        )
        final_url = sol.get("url") or target_url
        headers = sol.get("headers") or {}
        ctype = str(headers.get("content-type", "")).lower()

        is_pdf_like = looks_pdf_from_headers_url(
            ctype, final_url
        ) or _is_probable_pdf_url(final_url)
        if force_download or is_pdf_like:
            ctx = _ctx_from_fs_solution(
                sol, final_url if not force_download else target_url
            )
            try:
                if force_download:
                    page = ctx.new_page()
                    try:
                        return _download_via_goto(page, target_url)
                    finally:
                        try:
                            page.close()
                        except PlaywrightError:
                            pass
                else:
                    resp = ctx.request.get(final_url, timeout=NAV_TIMEOUT_MS)
                    blob = resp.body()
                    return (
                        parse_pdf_bytes(blob, max_pages=PDF_MAX_PAGES),
                        final_url,
                    )
            finally:
                try:
                    ctx.close()
                except PlaywrightError:
                    pass

        html = sol.get("response") or ""
        return extract_text(html, base_url=final_url), final_url

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
        page = context.new_page()
        try:
            # Handle obvious file URLs first with normal path
            if _is_probable_pdf_url(target_url):
                try:
                    return _download_via_goto(page, target_url)
                except PlaywrightError:
                    if fs_client:
                        logger.debug(
                            "download via goto failed; attempting FlareSolverr-assisted download"
                        )
                        return _fs_fetch(target_url)
                    raise

            logger.debug(f"goto domcontentloaded: {target_url}")
            response = page.goto(
                target_url,
                wait_until="domcontentloaded",
                timeout=NAV_TIMEOUT_MS,
            )

            # If response indicates block or the page body looks like a block, fall back to FlareSolverr.
            if _should_use_flaresolverr(
                target_url, page=page, response=response
            ):
                logger.debug(
                    "soft block or 403/429 detected; using FlareSolverr fallback"
                )
                return _fs_fetch(target_url)
            else:
                # Normal extraction path
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
                    logger.debug(f"download via goto failed: {e2}")
                    if fs_client:
                        logger.debug(
                            "attempting FlareSolverr-assisted download after goto failure"
                        )
                        return _fs_fetch(target_url)
                    # Fallback to request client as last resort
                    logger.debug("falling back to request client without FS")
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
            # Other Playwright errors: try FlareSolverr as a last resort
            if fs_client:
                logger.debug(
                    f"Playwright error encountered; trying FlareSolverr: {e}"
                )
                try:
                    return _fs_fetch(target_url)
                except Exception as e2:
                    logger.debug(f"FlareSolverr fallback failed: {e2}")
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
                        f"this is the final url {final_url} and text content {text_content[:200]}..."
                    )
                    url_to_content_queue.put(
                        (link_id, target_url, text_content)
                    )
                    processed_count += 1
                    logger.info(
                        f"queued result for {target_url} (total processed: {processed_count})"
                    )
                except PlaywrightError as e:
                    err_txt = (
                        f"PlaywrightError on {target_url}: {e}\n"
                        + "".join(
                            traceback.format_exception(
                                type(e), e, e.__traceback__
                            )
                        )
                    )
                    # truncate to avoid gigantic rows
                    err_txt = err_txt[:2000]
                    try:
                        with SessionLocal() as sess:
                            link = sess.get(Link, link_id)
                            if link:
                                link.fetch_error = err_txt
                                sess.commit()
                    except Exception as _db_err:
                        logger.warning(
                            f"failed to persist fetch_error for link_id={link_id}: {_db_err}"
                        )

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
                    err_txt = (
                        f"Unhandled scrape error on {target_url}: {e}\n"
                        + "".join(
                            traceback.format_exception(
                                type(e), e, e.__traceback__
                            )
                        )
                    )
                    err_txt = err_txt[:2000]
                    try:
                        with SessionLocal() as sess:
                            link = sess.get(Link, link_id)
                            if link:
                                link.fetch_error = err_txt
                                sess.commit()
                    except Exception as _db_err:
                        logger.warning(
                            f"failed to persist fetch_error for link_id={link_id}: {_db_err}"
                        )

                    logger.exception(
                        f"unhandled scrape error for {target_url}: {e}"
                    )
                    # stop_processing_event.set()
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
                if fs_client:
                    fs_client.destroy_all()
            except Exception:
                logger.debug("ignoring FlareSolverr cleanup error")
            try:
                if browser:
                    browser.close()
            except PlaywrightError as e:
                logger.debug(f"ignoring browser close error: {e}")
            logger.info(f"worker exit after processing {processed_count} pages")


@catch_and_log()
def scrape_urls(
    url_generator: Iterable[str],
    n_workers: int,
    manager: Manager,
    log_queue: MPQueue,
    listener: logging.handlers.QueueListener,
):
    """Driver to launch workers to scrape the urls in the generator."""
    url_link_tuples_process_queue = manager.Queue()
    url_to_content_queue = manager.Queue()
    stop_processing_event = manager.Event()
    flaresolver_semaphore = manager.Semaphore(n_workers)
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "scrape_urls")
    try:
        logger.info("start scrape_workers")
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


import os


@catch_and_log()
def stream_links(
    log_queue: MPQueue,
    max_items: int = None,
    debug_url: str = None,
) -> Iterator[Tuple[int, str]]:
    def _is_error_like(text: str | None) -> bool:
        if text is None:
            return True
        return False
        t = text.strip()
        if not t:
            return True
        if t.startswith("Error:"):
            return True
        if "Traceback" in t:
            return True
        if "HTTP 4" in t or "HTTP 5" in t:
            return True
        return False

    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "stream_links"
    )

    if debug_url:
        logger.info(f"debug mode: yielding only url={debug_url!r}")
        with SessionLocal() as sess:
            stmt = (
                select(Link.id, Link.url, Link.fetch_error, Content.text)
                .outerjoin(Content, Link.content_id == Content.id)
                .where(Link.url == debug_url)
                .execution_options(stream_results=True, yield_per=1)
            )
            row = sess.execute(stmt).first()
            if row is None:
                logger.warning(f"no row found for debug url {debug_url!r}")
            else:
                link_id, url, fetch_error, text = row
                if fetch_error is not None:
                    logger.warning(
                        f"debug url has fetch_error; skipping: {fetch_error}"
                    )
                elif _is_error_like(text):
                    logger.debug(
                        f"yielding debug id={link_id}, url={url}, content_is_none={text is None}"
                    )
                    yield (link_id, url)
                else:
                    logger.info(
                        "debug url content is not error-like; nothing to yield"
                    )
        logger.info("stream_links terminated (debug mode)")
        return

    logger.info("open Session")
    with SessionLocal() as sess:
        logger.info("build statement (links + content)")
        stmt = (
            select(Link.id, Link.url, Link.fetch_error, Content.text)
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
            link_id, url, fetch_error, text = row
            # if fetch_error is not None :
            #     continue
            if _is_error_like(text):
                yielded += 1
                if yielded % 1000 == 0:
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
    logger.info("stream_links terminated")


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _utf8_sanitize(s: str) -> str:
    # strip unpaired surrogates, then ensure utf-8 encodable
    if not isinstance(s, str):
        s = str(s)
    s = _SURROGATE_RE.sub("", s)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


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
                            content.encode("utf-8", errors="ignore")
                        ).hexdigest()

                        # insert the new content if it does not exist
                        is_valid = None
                        if guess_if_error_text(content):
                            is_valid = False
                        sess.execute(
                            sqlite_insert(Content)
                            .values(
                                content_hash=content_hash,
                                text=_utf8_sanitize(content),
                                is_valid=is_valid,
                            )
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
                        logger.info(f"{link_id}: {url}: got content")
                    sess.commit()
                except Exception:
                    logger.exception(
                        f"something bad happened on insert for {url}"
                    )
                    sess.rollback()
                    raise
                finally:
                    pending.clear()
    logger.info("db_writer terminated")


async def _main():
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/scraper_errors.log"
    )

    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(logger)
    max_items = None
    debug_url = None  # "https://www.researchgate.net/figure/Composition-of-Tree-Sparrow-nestling-diet-at-sites-throughout-the-UK-a-Variation-of_fig1_278239756"
    max_concurrent = 24
    scrape_n_concurrent = 1 if debug_url or max_items else max_concurrent
    logger.info(scrape_n_concurrent)
    n_workers = min(scrape_n_concurrent, psutil.cpu_count(logical=False))
    scrape_urls(
        stream_links(log_queue, max_items=max_items, debug_url=debug_url),
        n_workers,
        manager,
        log_queue,
        listener,
    )


if __name__ == "__main__":
    asyncio.run(_main())
