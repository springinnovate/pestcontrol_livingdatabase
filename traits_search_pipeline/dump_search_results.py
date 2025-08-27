from datetime import datetime
import json

from models import SearchResult, get_session
from sqlalchemy import (
    select,
)
from sqlalchemy.orm import Session


def dump_search_results(
    target_path: str, concurrency: int, max_visited: int
) -> None:
    session = get_session()
    urls = []

    seen = set()

    stmt = select(SearchResult.links)
    for links in session.execute(stmt):
        if not links:
            continue
        for url_list in links:
            for url in url_list:
                if url not in seen:
                    urls.append(url)
    payload = {
        "concurrency": concurrency,
        "max_visited": max_visited,
        "url": "blank",
        "urls": urls,
    }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    target_path = datetime.now().strftime(
        "dump_search_results_%Y_%m_%d_%H_%M_%S.json"
    )
    dump_search_results(target_path, 50, 1)
