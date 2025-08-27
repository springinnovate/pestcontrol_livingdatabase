from typing import List, Dict, Union, NamedTuple
import asyncio
import logging
import sys
import httpx

from tqdm import tqdm
from sqlalchemy import (
    insert,
    select,
)
from sqlalchemy.orm import Session

from models import SearchResult, get_session

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_species_list(filename: str):
    """Load the species names from a text file."""
    with open(filename, "r") as file:
        return [line.strip() for line in file.readlines()]


def expand_questions(
    questions: List[str],
    species_dict: Dict[str, Union[List[str], Dict[str, str]]],
):
    """Expand questions by replacing placeholders with species names."""
    expanded_questions = []

    for question in questions:
        for placeholder, species_list in species_dict.items():
            if f"[{placeholder}]" in question:
                for species in species_list:
                    expanded_question = question.replace(f"[{placeholder}]", species)
                    expanded_questions.append(
                        expanded_question.split(" | ") + [species]
                    )
    return expanded_questions


MAX_CONCURRENT = 10  # simultaneous requests
QPS = 20  # queries per second (overall throttle)


class SEOSearchResult(NamedTuple):
    url: str
    title: str


async def query(keyword: str, username: str, password: str) -> List[SEOSearchResult]:
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

    payload = [
        {
            "language_name": "English",
            "location_name": "United States",
            "keyword": keyword,
            "depth": 10,
        }
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, auth=(username, password), json=payload)
        r.raise_for_status()
        data = r.json()
        items = data.get("tasks", [{}])[0].get("result", [{}])[0].get("items", [])

        if not items:
            return []

        return [
            SEOSearchResult(url=i.get("url", ""), title=i.get("title", ""))
            for i in items
            if i.get("type") == "organic"
        ]


def insert_result(
    session: Session,
    question: str,
    keyword_query: str,
    species_name: str,
    links: List[str],
):
    stmt = insert(SearchResult).values(
        question=question,
        species_name=species_name,
        keyword_query=keyword_query,
        links=links,
    )
    session.execute(stmt)
    session.commit()


def question_exists(session: Session, question: str) -> bool:
    stmt = select(SearchResult).where(SearchResult.question == question)
    return session.execute(stmt).first() is not None


async def main():
    LOGGER.info("loading questions")
    question_template_list = open("data/question_list.txt").read().strip().split("\n")
    species_dict = {
        "predator_list": load_species_list("data/predator_list.txt"),
        "pest_list": load_species_list("data/pest_list.txt"),
        "full_species_list": load_species_list("data/full_species_list.txt"),
    }
    keywords_list, expanded_question_list, species_list = zip(
        *(expand_questions(question_template_list, species_dict))
    )

    # Configure HTTP basic authorization: basicAuth
    LOGGER.info("load username/password")
    username, password = open("secrets/auth").read().strip().split("\n")
    LOGGER.info("launching client")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    delay = 1 / QPS

    session = get_session()

    pending = [
        (k, q, s)
        for k, q, s in zip(keywords_list, expanded_question_list, species_list)
        if not question_exists(session, q)
    ]
    print(len(pending))

    async def rate_limited_query(
        question,
        keyword_query,
        species_name,
    ):
        async with sem:
            try:
                query_results = await query(keyword_query, username, password)
                url_list = []
                if query_results:
                    url_list = [q.url for q in query_results]
                insert_result(
                    session,
                    question,
                    keyword_query,
                    species_name,
                    url_list,
                )
                return [q.url for q in query_results]
            except Exception as e:
                print(e)
            await asyncio.sleep(delay)

    LOGGER.info("awaiting results")
    tasks = [
        asyncio.create_task(rate_limited_query(question, keyword_query, species))
        for keyword_query, question, species in pending
    ]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="queries"):
        await fut

    LOGGER.info("all done!")


if __name__ == "__main__":
    asyncio.run(main())
