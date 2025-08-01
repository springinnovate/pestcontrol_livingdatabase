print("starting module")
from typing import List, Dict, Union
import asyncio
import logging
import sys

print("loading api client")
from dataforseo_client import Configuration, ApiClient

print("loading onpageapi")
from dataforseo_client.api.on_page_api import OnPageApi

print("SerpApi")
from dataforseo_client.api.serp_api import SerpApi

print("ApiException")
from dataforseo_client.exceptions import ApiException

print("SerpGoogleOrganicLiveAdvancedRequestInfo")
from dataforseo_client.models import SerpGoogleOrganicLiveAdvancedRequestInfo

print("OnPageContentParsingLiveRequest")
from dataforseo_client.models import OnPageContentParsingLiveRequest


print("continuing...")
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


async def crawl_url(url: str, configuration: Configuration):
    with ApiClient(configuration) as client:
        api = OnPageApi(client)
        try:
            response = api.content_parsing_live(
                [OnPageContentParsingLiveRequest(url=url, markdown_view=True)]
            )
            result = response[0]
            markdown = result.page_as_markdown
            print(markdown)
        except ApiException as e:
            print(f"Error: {e}")


async def query(keyword: str, api: SerpApi):
    req = [
        SerpGoogleOrganicLiveAdvancedRequestInfo(
            language_name="English",
            location_name="United States",
            keyword=keyword,
        )
    ]
    return await asyncio.to_thread(api.google_organic_live_advanced, req)


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
    configuration = Configuration(username=username, password=password)

    LOGGER.info("launching client")
    with ApiClient(configuration) as client:
        api = SerpApi(client)
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
        results = await asyncio.gather(
            *(limited_query(k) for k in keywords_list)
        )
        LOGGER.info("done with results")
    for result in results:
        print(result.url)
        page = await crawl_url(result.url, configuration)
        print(page)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
