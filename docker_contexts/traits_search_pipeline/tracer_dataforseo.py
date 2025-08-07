from typing import List, Dict, Union, NamedTuple
import asyncio
import logging
import requests
import sys

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)

LOGGER = logging.getLogger(__name__)


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
                    expanded_question = question.replace(
                        f"[{placeholder}]", species
                    )
                    expanded_questions.append(expanded_question.split(" | "))
    return expanded_questions


MAX_CONCURRENT = 10  # simultaneous requests
QPS = 20  # queries per second (overall throttle)


class SearchResult(NamedTuple):
    url: str
    title: str


async def query(
    keyword: str, username: str, password: str
) -> List[SearchResult]:
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

    payload = [
        {
            "language_name": "English",
            "location_name": "United States",
            "keyword": keyword,
        }
    ]
    response = requests.post(url, auth=(username, password), json=payload)
    results: List[SearchResult] = []
    if response.ok:
        data = response.json()
        # we can only submit one task so we always ["tasks"][0]
        # and there is always one result, so ["result"][0]
        items = data["tasks"][0]["result"][0]["items"]
        for item in items:
            if item["type"] == "organic":
                result = SearchResult(
                    url=item.get("url", ""), title=item.get("title", "")
                )
                results.append(result)
        return results
    else:
        print(f"Request failed: {response.status_code}, {response.text}")


async def crawl_url(url: str, username: str, password: str):
    api_url = "https://api.dataforseo.com/v3/on_page/content_parsing/live"

    payload = [{"url": url, "markdown_view": True}]

    response = requests.post(api_url, json=payload, auth=(username, password))

    if response.ok:
        result = response.json()
        LOGGER.debug(result)
        markdown = result["tasks"][0]["result"][0]["page_as_markdown"]
        print(markdown)
    else:
        print(f"Error: {response.status_code} - {response.text}")


async def main():
    LOGGER.info("loading questions")
    questions_list = open("data/question_list.txt").read().strip().split("\n")
    species_dict = {
        "predator_list": load_species_list("data/predator_list.txt"),
        "pest_list": load_species_list("data/pest_list.txt"),
        "full_species_list": load_species_list("data/full_species_list.txt"),
    }
    keywords_list, question_list = zip(
        *(expand_questions(questions_list, species_dict)[:1])
    )

    # Configure HTTP basic authorization: basicAuth
    LOGGER.info("load username/password")
    username, password = open("secrets/auth").read().strip().split("\n")
    query_results = await query(keywords_list[0], username, password)
    for query_result in query_results:
        LOGGER.info(query_result)
        crawl_result = await crawl_url(query_result.url, username, password)
        LOGGER.debug(crawl_result)
    return
    LOGGER.info("launching client")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    delay = 1 / QPS

    async def limited_query(kw):
        async with sem:
            try:
                res = await query(kw, api)
                print(res)  # process response here
            except ApiException as e:
                print(e)
            await asyncio.sleep(delay)

    LOGGER.info("awaiting results")
    results = await asyncio.gather(*(limited_query(k) for k in keywords_list))
    LOGGER.info("done with results")
    for result in results:
        print(result.url)
        page = await crawl_url(result.url, username, password)
        print(page)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
