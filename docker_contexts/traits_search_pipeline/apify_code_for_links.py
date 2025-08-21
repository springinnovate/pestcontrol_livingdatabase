from __future__ import annotations

import asyncio
import re
import html as html_utils
from urllib.parse import urlsplit, urlunsplit
from io import BytesIO

import httpx
from apify import Actor
from bs4 import BeautifulSoup

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

URL_TO_TEXT: dict[str, str] = {}


def extract_text(html: str, base_url: str | None = None) -> str:
    try:
        import trafilatura

        txt = trafilatura.extract(
            html, url=base_url, include_comments=False, include_tables=False
        )
        if txt and txt.strip():
            return txt.strip()
    except Exception:
        pass
    try:
        from readability import Document  # readability-lxml

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


def site_origin(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "", "", ""))


def parent_url(url: str) -> str:
    p = urlsplit(url)
    parent = p.path.rsplit("/", 1)[0] or "/"
    return urlunsplit((p.scheme, p.netloc, parent, "", ""))


def browser_like_headers(
    url: str, referer: str | None = None
) -> dict[str, str]:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }
    if referer:
        headers["Referer"] = referer
    if url.lower().endswith(".pdf"):
        headers["Accept"] = "application/pdf,*/*;q=0.8"
    return headers


async def playwright_fetch_bytes(
    url: str, referer: str | None = None
) -> bytes | None:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=DEFAULT_UA)
        page = await context.new_page()
        if referer:
            try:
                await page.goto(
                    referer, wait_until="domcontentloaded", timeout=30_000
                )
            except Exception:
                pass
        try:
            resp = await context.request.get(
                url,
                headers=browser_like_headers(url, referer=referer),
                timeout=90_000,
            )
            if resp.ok:
                data = await resp.body()
                await browser.close()
                return data
        except Exception:
            pass
        await browser.close()
    return None


def looks_pdf(content_type: str, url: str) -> bool:
    return "application/pdf" in content_type or url.lower().endswith(".pdf")


def parse_pdf_bytes(data: bytes, max_pages: int = 10) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        n_pages = len(reader.pages)
        if n_pages > max_pages:
            return f"[[PDF_SKIPPED_TOO_LONG:{n_pages}]]"
        texts: list[str] = []
        for i in range(n_pages):
            page_text = (reader.pages[i].extract_text() or "").strip()
            if page_text:
                texts.append(page_text)
        return "\n\n".join(texts) if texts else "[[PDF_EMPTY_TEXT]]"
    except Exception as e:
        return f"[[PDF_PARSE_ERROR:{e.__class__.__name__}: {e}]]"


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    *,
    referer_hint: str | None = None,
    allow_playwright_fallback: bool = False,
) -> str | None:
    try:
        Actor.log.info(f"Fetching URL: {url}")
        origin_url = site_origin(url)
        parent = parent_url(url)

        # best-effort priming
        for u in (origin_url, referer_hint):
            if not u:
                continue
            try:
                await client.get(u, headers=browser_like_headers(u))
            except Exception:
                pass

        # attempt with a sequence of referers, stopping on first non-403
        referers = [referer_hint or origin_url, parent]
        if referer_hint:
            referers.append(referer_hint)
        unique_referers: list[str] = []
        for r in referers:
            if r and r not in unique_referers:
                unique_referers.append(r)

        response: httpx.Response | None = None
        for ref in unique_referers:
            response = await client.get(
                url, headers=browser_like_headers(url, referer=ref)
            )
            if response.status_code != 403:
                break

        if response is None:
            return "Error: no response received"

        content_type = response.headers.get("content-type", "").lower()

        # 403 PDF fallback via Playwright (optional)
        if (
            response.status_code == 403
            and looks_pdf(content_type, url)
            and allow_playwright_fallback
        ):
            data = await playwright_fetch_bytes(
                url, referer=referer_hint or parent
            )
            if data:
                return parse_pdf_bytes(data, max_pages=10)
            return "Error: 403 and Playwright fallback failed to fetch bytes"

        response.raise_for_status()

        if looks_pdf(content_type, url):
            return parse_pdf_bytes(response.content, max_pages=10)
        if "text/plain" in content_type:
            return response.text
        if "html" in content_type or not content_type:
            return extract_text(response.text, base_url=str(response.url))
        return f"Unsupported content type: {content_type}"

    except httpx.TimeoutException:
        return f"Error: timed out while fetching {url}"
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        msg = f"Error: server returned status {code} for {url}"
        if code == 403:
            msg += (
                " — likely hotlink protection or a cookie/Referer requirement."
            )
        return msg
    except httpx.RequestError as e:
        return f"Error: request failed for {url} ({e.__class__.__name__})"
    except Exception as e:
        return f"Error: unexpected failure fetching {url} ({e})"


async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    allow_playwright_fallback: bool,
):
    while True:
        url = await queue.get()
        try:
            if url is None:
                Actor.log.info(f"[w{worker_id}] received sentinel; exiting")
                break
            result = await fetch_url(
                client, url, allow_playwright_fallback=allow_playwright_fallback
            )
            if result is not None:
                URL_TO_TEXT[url] = result
        except Exception:
            Actor.log.exception(f"[w{worker_id}] error processing {url!r}")
        finally:
            queue.task_done()


async def main() -> None:
    async with Actor:
        inp = await Actor.get_input() or {}
        urls = inp.get("urls", [])
        max_urls = inp.get("max_urls", None)
        if isinstance(max_urls, int) and max_urls and max_urls > 0:
            urls = urls[:max_urls]
        concurrency = max(1, int(inp.get("concurrency", 1)))
        use_playwright = bool(inp.get("use_playwright", False))

        if not urls:
            raise ValueError('Input must contain a "urls" array.')

        queue: asyncio.Queue = asyncio.Queue()
        for url in urls:
            queue.put_nowait(url)
        num_workers = min(concurrency, len(urls))
        for _ in range(num_workers):
            queue.put_nowait(None)

        proxy_cfg = await Actor.create_proxy_configuration(groups=["auto"])
        proxy_url = await proxy_cfg.new_url()

        transport = httpx.AsyncHTTPTransport(proxy=proxy_url)
        async with httpx.AsyncClient(
            transport=transport,
            http2=True,
            timeout=httpx.Timeout(30.0),
            headers=browser_like_headers(""),
            follow_redirects=True,
        ) as client:
            tasks = [
                asyncio.create_task(
                    worker(wid + 1, queue, client, use_playwright)
                )
                for wid in range(num_workers)
            ]
            await queue.join()
            await asyncio.gather(*tasks, return_exceptions=True)

        await Actor.set_value("OUTPUT", URL_TO_TEXT)
