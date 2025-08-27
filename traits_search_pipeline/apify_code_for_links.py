"""
docker build -t apify_pcld:latest . && docker run --rm -it -v "%CD%/apify_storage":/apify_storage apify_pcld:latest

"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from io import BytesIO
from typing import Iterable
import html as html_utils
import sys

import trafilatura
from readability import Document  # readability-lxml

from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    TimeoutError as PWTimeoutError,
)
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pm_extract

LOGGER = logging.getLogger("fetcher")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


def extract_text(html: str, base_url: str | None = None) -> str:
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

    scrubbed_html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
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


# -----------------------------------
# core fetch
# -----------------------------------
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

        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
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


async def fetch_url(url: str, *, timeout_ms: int = 60000, max_retries: int = 3) -> str:
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
                if text and not await soft_block_detect(await context.new_page()):
                    return text
                LOGGER.info("retry %d for %s", attempt, url)
                await asyncio.sleep(0.8 + random.random() * 1.2)
            return text  # type: ignore[has-type]
        finally:
            await context.close()
            await browser.close()


# -----------------------------------
# worker pool
# -----------------------------------
async def worker(worker_id: int, q: asyncio.Queue[str], out: dict[str, str], ctx):
    LOGGER.info("[w%d] start", worker_id)
    while True:
        url = await q.get()
        if url is None:  # type: ignore[unreachable]
            q.task_done()
            break
        try:
            page = await ctx.new_page()
            try:
                # reuse the context per worker for speed
                resp = await page.goto(
                    url, wait_until="domcontentloaded", timeout=60000
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=60000)
                except PWTimeoutError:
                    pass
                ctype = (resp.headers.get("content-type") if resp else "") or ""
                if looks_pdf_from_headers_url(ctype, resp.url if resp else url):
                    data = await (resp.body() if resp else b"")
                    out[url] = parse_pdf_bytes(data, max_pages=20)
                else:
                    html = await page.content()
                    out[url] = extract_text(html, base_url=(resp.url if resp else url))
                LOGGER.info("[w%d] ok: %s", worker_id, url)
            finally:
                await page.close()
        except Exception as e:
            LOGGER.info("[w%d] err: %s -> %s", worker_id, url, e)
            out[url] = f"Error: {e}"
        finally:
            q.task_done()


async def run(urls: Iterable[str], concurrency: int = 4) -> dict[str, str]:
    urls = [u for u in urls if u]
    out: dict[str, str] = {}
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
        # create one context per worker
        contexts = [
            await browser.new_context(
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
            for _ in range(min(concurrency, max(1, len(urls))))
        ]

        q: asyncio.Queue[str] = asyncio.Queue()
        for u in urls:
            q.put_nowait(u)

        tasks = [
            asyncio.create_task(worker(i + 1, q, out, contexts[i]))
            for i in range(len(contexts))
        ]

        await q.join()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for ctx in contexts:
            await ctx.close()
        await browser.close()
    return out


# -----------------------------------
# example CLI usage
# -----------------------------------
if __name__ == "__main__":

    async def _main():
        if len(sys.argv) > 1:
            urls = sys.argv[1:]
        else:
            # quick demo set; replace as needed
            urls = ["https://example.com"]
        res = await run(urls, concurrency=4)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    asyncio.run(_main())
