#!/usr/bin/env python3
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, NamedTuple

import httpx
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from models import (
    DB_URI,
    Question,
    KeywordQuery,
    QuestionKeyword,
    SearchHead,
    Link,
    SearchResultLink,
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


async def query(
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


def ensure_search_result_link(
    session, search_head_id: int, link_id: int
) -> None:
    exists = session.scalar(
        select(func.count())
        .select_from(SearchResultLink)
        .where(
            SearchResultLink.search_head_id == search_head_id,
            SearchResultLink.link_id == link_id,
        )
    )
    if not exists:
        session.add(
            SearchResultLink(search_head_id=search_head_id, link_id=link_id)
        )


def fetch_pending(
    session, limit: int | None = None
) -> list[tuple[int, str, str]]:
    stmt = (
        select(
            SearchHead.question_id.label("search_head_id"),
            KeywordQuery.query.label("keyword"),
            Question.text.label("question_text"),
        )
        .join(Question, Question.id == SearchHead.question_id)
        .join(QuestionKeyword, QuestionKeyword.question_id == Question.id)
        .join(KeywordQuery, KeywordQuery.id == QuestionKeyword.keyword_query_id)
        .outerjoin(
            SearchResultLink,
            SearchResultLink.search_head_id == SearchHead.question_id,
        )
        .where(SearchResultLink.search_head_id.is_(None))
        .order_by(SearchHead.question_id.asc())
    )
    if limit:
        stmt = stmt.limit(limit)
    return [(sid, kw, qtxt) for sid, kw, qtxt in session.execute(stmt).all()]


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to the database, just print actions",
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
        search_head_id, keyword, question_text = item
        async with sem:
            try:
                results = await query(keyword, username, password)
            finally:
                if delay:
                    await asyncio.sleep(delay)

        with Session() as session:
            urls = [r.url for r in results]
            link_ids = [get_or_create_link(session, u) for u in urls]
            for lid in link_ids:
                ensure_search_result_link(session, search_head_id, lid)
            session.commit()
            LOGGER.info(
                "updated search_head=%s keyword=%r links_added=%d",
                search_head_id,
                keyword,
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
