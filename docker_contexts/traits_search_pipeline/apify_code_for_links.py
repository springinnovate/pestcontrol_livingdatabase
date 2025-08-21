from __future__ import annotations

import asyncio
import re
import html as html_utils

from apify import Actor
from bs4 import BeautifulSoup
from httpx import AsyncClient, AsyncHTTPTransport
from readability import Document

HEADERS = {"User-Agent": "Mozilla/5.0"}

URL_TO_TEXT: dict[str, str] = dict()


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
    # strip any code that passes through
    html2 = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    # replace any HTML headings with a newline
    html2 = re.sub(r"(?i)</?(p|div|br|li|h[1-6]|tr|th|td)\b[^>]*>", "\n", html2)
    # strip the rest of the html brackets out
    text = re.sub(r"(?s)<[^>]+>", " ", html2)
    text = html_utils.unescape(text)
    # scrub any intermediate whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\f\v]+", " ", text))
    return text.strip()


async def fetch_url(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        Actor.log.info(f"Fetching URL: {url}")
        resp = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        if "text/plain" in ct:
            return resp.text
        if "html" in ct or not ct:
            return extract_text(resp.text, base_url=str(resp.url))
        return f"Unsupported content type: {ct}"
    except httpx.TimeoutException:
        return f"Error: timed out while fetching {url}"
    except httpx.HTTPStatusError as e:
        return (
            f"Error: server returned status {e.response.status_code} for {url}"
        )
    except httpx.RequestError as e:
        return f"Error: request failed for {url} ({e.__class__.__name__})"
    except Exception as e:
        return f"Error: unexpected failure fetching {url} ({e})"


async def worker(worker_id: int, q: asyncio.Queue, client: AsyncClient):
    while True:
        url = await q.get()
        try:
            if url is None:  # sentinel for shutdown
                Actor.log.info(f"[w{worker_id}] received sentinel; exiting")
                break
            html = await fetch_url(client, url)
            if html is not None:
                URL_TO_TEXT[url] = html
        except Exception:
            Actor.log.exception(f"[w{worker_id}] error processing {url!r}")
        finally:
            q.task_done()


async def main() -> None:
    global MAX_VISITED, CONCURRENCY
    async with Actor:
        inp = await Actor.get_input() or {}
        max_urls = inp.get("max_urls", -1)
        urls = inp.get("urls", [])[:max_urls]
        concurrency = inp.get("concurrency", 1)

        if not urls:
            raise ValueError('Input must contain a "urls" array.')

        queue = asyncio.Queue()
        for url in urls:
            queue.put_nowait(url)
        n_workers = min(concurrency, len(urls))

        for _ in range(n_workers):
            queue.put_nowait(None)

        proxy_cfg = await Actor.create_proxy_configuration(groups=["auto"])
        proxy_url = await proxy_cfg.new_url()

        transport = AsyncHTTPTransport(proxy=proxy_url)
        async with AsyncClient(transport=transport) as client:
            tasks = [
                asyncio.create_task(worker(worker_id + 1, queue, client))
                for worker_id in range(n_workers)
            ]
            await queue.join()
            await asyncio.gather(*tasks, return_exceptions=True)

        print(f"set output to {URL_TO_TEXT}")
        await Actor.set_value("OUTPUT", URL_TO_TEXT)
