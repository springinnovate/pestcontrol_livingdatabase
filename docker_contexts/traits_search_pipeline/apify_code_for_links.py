from __future__ import annotations

import asyncio
import traceback

from apify import Actor
from httpx import AsyncClient, AsyncHTTPTransport

HEADERS = {"User-Agent": "Mozilla/5.0"}

URL_TO_TEXT: dict[str, str] = dict()


async def fetch_url(client: AsyncClient, url: str) -> str | None:
    try:
        Actor.log.info(f"Fetching URL: {url}")
        response = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        tb = traceback.format_exc()
        Actor.log.warning(f"Fetch failed for {url}: {e}\n{tb}")
        raise


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
