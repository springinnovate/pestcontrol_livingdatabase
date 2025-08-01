from typing import List, Dict, Union
import asyncio
import logging
import sys

from dataforseo_client import Configuration, ApiClient
from dataforseo_client.api.serp_api import SerpApi
from dataforseo_client.exceptions import ApiException
from dataforseo_client.models import SerpGoogleOrganicLiveAdvancedRequestInfo


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
    configuration = Configuration(
        username="richard.sharp@wwfus.org", password="99b21285fea6911f"
    )

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

        results = await asyncio.gather(
            *(limited_query(k) for k in keywords_list)
        )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
