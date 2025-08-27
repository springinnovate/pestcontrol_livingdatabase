"""docker build -t apify_pcld:latest . && docker run --rm -it -e CRAWLEE_PURGE_ON_START=0 -e APIFY_PURGE_ON_START=0 -e APIFY_LOCAL_STORAGE_DIR=/apify_storage -v "%CD%/apify_storage":/apify_storage apify_pcld:latest"""

from __future__ import annotations

import asyncio
import html as html_utils
import logging
import re
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit, unquote

from apify import Actor
from bs4 import BeautifulSoup
from diskcache import Index
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from pypdf import PdfReader
from readability import Document as ReadabilityDocument
import httpx
import trafilatura

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BROWSER_CACHE = Index("browser_cache")
LOGGER = logging.getLogger(__name__)


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _title_from_researchgate_url(url: str) -> str | None:
    # e.g. .../publication/259592301_Flowers_for_better_pest_control_Ground_cover_plants...
    m = re.search(r"/publication/\d+_([^/?#]+)", url)
    if not m:
        return None
    slug = unquote(m.group(1))
    # tidy slug → title-ish
    title = re.sub(r"[_+]+", " ", slug).strip()
    # strip trailing id-like chunks in parentheses if present
    title = re.sub(r"\s*\(\d{4}\)$", "", title)
    return title if title else None


async def _crossref_find_doi_by_title(
    client: httpx.AsyncClient, title: str
) -> str | None:
    # Use query.bibliographic (title+authors+etc) and pick the top item
    params = {"query.bibliographic": title, "rows": 3}
    r = await client.get(
        "https://api.crossref.org/works", params=params, timeout=20
    )
    if r.status_code != 200:
        return None
    items = r.json().get("message", {}).get("items", []) or []
    for it in items:
        doi = it.get("DOI")
        if doi:
            return doi
    return None


async def _unpaywall_best_oa_from_doi(
    client: httpx.AsyncClient, doi: str, email: str | None
) -> str | None:
    if not email:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    r = await client.get(url, params={"email": email}, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    loc = data.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")


async def _openalex_best_oa_from_title(
    client: httpx.AsyncClient, title: str
) -> str | None:
    # Search by title and prefer best_oa_location.url
    params = {"search": title, "per_page": 5}
    r = await client.get(
        "https://api.openalex.org/works", params=params, timeout=20
    )
    if r.status_code != 200:
        return None
    results = (r.json() or {}).get("results", []) or []
    for it in results:
        loc = it.get("best_oa_location") or {}
        url = loc.get("url")
        if url:
            return url
        # fallback to primary_location if present
        pl = it.get("primary_location") or {}
        if pl.get("source", {}) and pl.get("source", {}).get("is_oa"):
            if pl.get("pdf_url") or pl.get("landing_page_url"):
                return pl.get("pdf_url") or pl.get("landing_page_url")
    return None


async def _wayback_available(client: httpx.AsyncClient, url: str) -> str | None:
    # Returns the archived snapshot URL if available
    r = await client.get(
        "https://archive.org/wayback/available", params={"url": url}, timeout=15
    )
    if r.status_code != 200:
        return None
    snap = (r.json() or {}).get("archived_snapshots", {})
    closest = snap.get("closest") or {}
    if closest.get("available") and closest.get("url"):
        return closest["url"]
    return None


async def resolve_open_version_for_researchgate(
    client: httpx.AsyncClient, url: str
) -> str | None:
    title = _title_from_researchgate_url(url)
    if not title:
        # Try Wayback on the RG URL directly (sometimes archived)
        return await _wayback_available(client, url)

    # 1) Crossref → DOI
    doi = await _crossref_find_doi_by_title(client, title)

    # 2) Unpaywall (needs email) if we have DOI
    if doi:
        oa = await _unpaywall_best_oa_from_doi(client, doi, None)
        if oa:
            return oa

    # 3) OpenAlex by title
    oa = await _openalex_best_oa_from_title(client, title)
    if oa:
        return oa

    # 4) Wayback on potential publisher/landing pages, if known DOI
    if doi:
        # heuristic publisher landing from doi
        return await _wayback_available(client, f"https://doi.org/{doi}")

    # 5) Wayback on RG itself as last resort
    return await _wayback_available(client, url)


def extract_readable_text(html: str) -> str:
    try:
        # trafilatura works best if it does work
        txt = trafilatura.extract(
            html, include_comments=False, include_tables=False
        )
        if txt and txt.strip():
            return txt.strip()
    except Exception:
        pass

    try:
        # TODO: what is readabiltydocument
        summary_html = ReadabilityDocument(html).summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        pass

    try:
        # this just tries to parse html
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        pass

    # regex scrub as a last resort
    scrubbed = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    scrubbed = re.sub(
        r"(?i)</?(p|div|br|li|h[1-6]|tr|th|td)\b[^>]*>", "\n", scrubbed
    )
    text = re.sub(r"(?s)<[^>]+>", " ", scrubbed)
    text = html_utils.unescape(text)
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\f\v]+", " ", text))
    return text.strip()


def parse_pdf_content(data: bytes, max_pages: int) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        n_pages = len(reader.pages)
        if n_pages > max_pages:
            return f"[[PDF_SKIPPED_TOO_LONG:{n_pages}]]"
        parts: list[str] = []
        for i in range(n_pages):
            t = (reader.pages[i].extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts) if parts else "[[PDF_EMPTY_TEXT]]"
    except Exception as e:
        return f"[[PDF_PARSE_ERROR:{e.__class__.__name__}: {e}]]"


def site_origin(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "", "", ""))


def parent_url(url: str) -> str:
    p = urlsplit(url)
    parent = p.path.rsplit("/", 1)[0] or "/"
    return urlunsplit((p.scheme, p.netloc, parent, "", ""))


def browser_like_headers(url: str, referer: str) -> dict[str, str]:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Sec-CH-UA": '"Chromium";v="124", "Not.A/Brand";v="24"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    }
    if referer and referer != url:
        headers["Referer"] = referer
        headers["Sec-Fetch-Site"] = "same-origin"
    if url.lower().endswith(".pdf"):
        headers["Accept"] = "application/pdf,*/*;q=0.8"
        headers["Sec-Fetch-Dest"] = "document"
    return headers


async def playwright_fetch_text(url: str, referer: str) -> str | None:
    if url in BROWSER_CACHE:
        return BROWSER_CACHE[url]

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=DEFAULT_UA,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        await Stealth().apply_stealth_async(context)

        try:
            resp = await context.request.get(
                url,
                headers=browser_like_headers(url, referer),
                timeout=90_000,
            )
            if not resp.ok:
                await browser.close()
                return f"Error: server returned status {resp.status} for {url}"
            ct = (resp.headers.get("content-type") or "").lower()
            body = await resp.body()
            if "application/pdf" in ct or url.lower().endswith(".pdf"):
                text = parse_pdf_content(body, max_pages=10)
            else:
                html = body.decode("utf-8", errors="ignore")
                text = extract_readable_text(html)
        except Exception as e:
            await browser.close()
            return f"Error: unexpected failure fetching {url} ({e})"

        await browser.close()
        BROWSER_CACHE[url] = text
        return text


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
) -> str | None:
    try:

        if _host(url).endswith("researchgate.net"):
            alt = await resolve_open_version_for_researchgate(client, url)
            if not alt:
                return "[[OA_NOT_FOUND]]"
            # fetch & parse the alt URL using your existing logic
            r = await client.get(
                alt,
                headers=browser_like_headers(alt, ""),
                timeout=30,
                follow_redirects=True,
            )
            r.raise_for_status()
            ct = (r.headers.get("content-type") or "").lower()
            if "pdf" in ct or alt.lower().endswith(".pdf"):
                return parse_pdf_content(r.content, max_pages=10)
            if "text/plain" in ct:
                return r.text
            if "html" in ct or not ct:
                return extract_readable_text(r.text, base_url=str(r.url))
            return f"Unsupported content type: {ct}"

        Actor.log.info(f"Fetching URL: {url}")
        origin_url = site_origin(url)
        ref_parent = parent_url(url)

        # best-effort priming to get set cookies
        try:
            await client.get(
                origin_url, headers=browser_like_headers(origin_url, "")
            )
        except Exception:
            pass

        referers = {origin_url, ref_parent}

        response: httpx.Response | None = None
        for ref in referers:
            r = await client.get(url, headers=browser_like_headers(url, ref))
            if r.status_code not in (401, 403):
                response = r
                break
            response = r  # remember last 401/403

        if response is None:
            return "Error: no response received"

        ct = (response.headers.get("content-type") or "").lower()

        # 401/403 → Playwright fallback if allowed
        if response.status_code in (401, 403):
            text = await playwright_fetch_text(url, ref_parent)
            return text

        response.raise_for_status()

        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            return parse_pdf_content(response.content, max_pages=20)
        if "text/plain" in ct:
            return response.text
        if "html" in ct or not ct:
            return extract_readable_text(response.text)
        return f"Unsupported content type: {ct}"

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
    metrics: dict[str, int],
):
    while True:
        url = await queue.get()
        try:
            if url is None:
                Actor.log.info(f"[w{worker_id}] received sentinel; exiting")
                break

            result = await fetch_url(client, url)
            ok = isinstance(result, str) and not result.startswith("Error:")
            await Actor.push_data({"url": url, "ok": ok, "content": result})

            metrics["ok" if ok else "err"] += 1

        except Exception as e:
            metrics["err"] += 1
            Actor.log.exception(f"[w{worker_id}] error processing {url!r}")
            await Actor.push_data(
                {"url": url, "ok": False, "content": f"Error: {e}"}
            )
        finally:
            queue.task_done()


async def make_client_for_worker(
    worker_id: int, proxy_cfg
) -> httpx.AsyncClient:
    if proxy_cfg:
        proxy_url = await proxy_cfg.new_url(session_id=f"worker_{worker_id}")
        transport = httpx.AsyncHTTPTransport(proxy=proxy_url)
    else:
        transport = httpx.AsyncHTTPTransport()

    limits = httpx.Limits(max_connections=50, max_keepalive_connections=25)
    default_headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    return httpx.AsyncClient(
        transport=transport,
        limits=limits,
        http2=True,
        timeout=httpx.Timeout(30.0),
        headers=default_headers,
        follow_redirects=True,
    )


async def main() -> None:
    async with Actor:
        inp = await Actor.get_input() or {}

        urls = inp.get("urls", [])
        max_urls = inp.get("max_urls")
        if isinstance(max_urls, int) and max_urls > 0:
            urls = urls[:max_urls]
        Actor.log.info(f'INPUT IS: {"empty" if not inp else len(urls)} urls')
        concurrency = max(1, int(inp.get("concurrency", 1)))

        if not urls:
            raise ValueError('Input must contain a "urls" array.')

        queue: asyncio.Queue = asyncio.Queue()
        for u in urls:
            queue.put_nowait(u)
        num_workers = min(concurrency, len(urls))
        for _ in range(num_workers):
            queue.put_nowait(None)

        proxy_cfg = None
        try:
            proxy_cfg = await Actor.create_proxy_configuration(groups=["auto"])
        except Exception:
            Actor.log.info("Apify Proxy unavailable; using direct connection.")

        metrics = {"ok": 0, "err": 0}

        clients = [
            await make_client_for_worker(i + 1, proxy_cfg)
            for i in range(num_workers)
        ]
        try:
            tasks = [
                asyncio.create_task(worker(i + 1, queue, clients[i], metrics))
                for i in range(num_workers)
            ]
            await queue.join()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await asyncio.gather(
                *(c.aclose() for c in clients), return_exceptions=True
            )

        # Small run summary (dataset has the per-URL items)
        await Actor.set_value(
            "SUMMARY",
            {
                "total": len(urls),
                "ok": metrics["ok"],
                "err": metrics["err"],
            },
        )
