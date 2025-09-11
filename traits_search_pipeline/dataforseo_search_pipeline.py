import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, NamedTuple

import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import (
    DB_URI,
    Question,
    Link,
    QuestionLink,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(funcName)s:%(lineno)d] %(message)s",
)
LOGGER = logging.getLogger("missing_links_fill")
logging.getLogger("httpx").setLevel(logging.WARNING)


class SEOSearchResult(NamedTuple):
    url: str
    title: str


async def run_serp_query(
    keyword: str, username: str, password: str, timeout: float = 30.0
) -> List[SEOSearchResult]:
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    payload = [
        {
            "language_name": "English",
            "location_name": "United States",
            "keyword": keyword,
            "depth": 10,
        }
    ]
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, auth=(username, password), json=payload)
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


def get_or_create_link(session, url: str) -> int:
    link_id = session.scalar(select(Link.id).where(Link.url == url))
    if link_id:
        return link_id
    obj = Link(url=url, content_id=None)
    session.add(obj)
    session.flush()
    return obj.id


def map_link_to_question(session, question_id: int, link_id: int) -> None:
    exists = session.scalar(
        select(QuestionLink.id).where(
            QuestionLink.question_id == question_id,
            QuestionLink.link_id == link_id,
        )
    )
    if exists is None:
        session.add(QuestionLink(question_id=question_id, link_id=link_id))


def fetch_pending(
    session, limit: int | None = None
) -> list[tuple[int, str, str]]:
    stmt = (
        select(
            Question.id.label("question_id"),
            Question.keyword_phrase.label("keyword"),
            Question.text.label("question_text"),
        )
        .outerjoin(QuestionLink, QuestionLink.question_id == Question.id)
        .where(QuestionLink.id.is_(None))
        .order_by(Question.id.asc())
    )
    if limit:
        stmt = stmt.limit(limit)
    return [(qid, kw, qtxt) for qid, kw, qtxt in session.execute(stmt).all()]


async def main():
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
    args = parser.parse_args()

    username, password = (
        args.auth_file.read_text(encoding="utf-8").strip().split("\n", 1)
    )

    engine = create_engine(args.db or DB_URI, future=True)
    Session = sessionmaker(engine, expire_on_commit=False)

    sem = asyncio.Semaphore(args.max_concurrent)
    delay = 1.0 / args.qps if args.qps > 0 else 0.0

    async def worker(item: tuple[int, str, str]):
        question_id, keyword_query, question_text = item
        async with sem:
            try:
                results = await run_serp_query(
                    keyword_query, username, password
                )
                print(results)
            finally:
                if delay:
                    await asyncio.sleep(delay)

        with Session() as session:
            urls = [r.url for r in results]
            link_ids = [get_or_create_link(session, u) for u in urls]
            for link_id in link_ids:
                map_link_to_question(session, question_id, link_id)
            session.commit()
            LOGGER.info(
                "updated search_head=%s keyword_query=%r links_added=%d",
                question_id,
                keyword_query,
                len(link_ids),
            )

    with Session() as session:
        pending = fetch_pending(session, args.limit)
        LOGGER.info("pending search_heads without links: %d", len(pending))

    tasks = [asyncio.create_task(worker(item)) for item in pending]
    for t in asyncio.as_completed(tasks):
        await t


if __name__ == "__main__":
    asyncio.run(main())
