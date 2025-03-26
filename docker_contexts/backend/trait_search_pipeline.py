import io
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse
import argparse
import asyncio
import collections
import hashlib
import json
import logging
import random
import re
import sys
import warnings

import aiohttp
import PyPDF2
from bs4 import BeautifulSoup
from diskcache import Cache
from openai import OpenAI
from playwright.async_api import async_playwright
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

SEARCH_CACHE = Cache('google_search_cache')
PROMPT_CACHE = Cache('openai_cache')
BROWSER_CACHE = Cache('browser_cache')
NLP_CACHE = Cache('nlp_cache')
# avoid searching the same domain at once
DOMAIN_SEMAPHORES = defaultdict(lambda: asyncio.Semaphore(1))
GLOBAL_SEMPAHORE = asyncio.Semaphore(1)

try:
    API_KEY = open('./secrets/custom_search_api_key.txt', 'r').read()
    CX = open('./secrets/cxid.txt', 'r').read()
    OPENAI_KEY = open('../../../LLMTechnicalWriter/.secrets/llm_grant_assistant_openai.key', 'r').read()
except FileNotFoundError:
    LOGGER.exception(
        'custom_search_api_key.txt API key not found, should be in '
        'subidrectory of "secrets"')


def encode_text(text, tokenizer, max_length=512):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return tokens


MAX_TABS = 10
MAX_OPENAI = 10
MAX_BACKOFF_WAIT = 8

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

for module in [
        'asyncio',
        'taskgraph',
        'selenium',
        'urllib3.connectionpool',
        'primp',
        'request',
        'httpcore',
        'openai',
        'httpx',]:
    logging.getLogger(module).setLevel(logging.ERROR)


def create_openai_context():
    openai_client = OpenAI(api_key=OPENAI_KEY)
    openai_context = {
        'client': openai_client,
    }
    return openai_context


async def google_custom_search_async(api_key, cx, query, num_results=25):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, google_custom_search_sync, api_key, cx, query, num_results
    )


def google_custom_search_sync(api_key, cx, query, num_results=25):
    cache_key = json.dumps({
        'api_key': api_key,
        'cx': cx,
        'query': query,
        'num_results': num_results
    }, sort_keys=True)

    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    url = 'https://www.googleapis.com/customsearch/v1'
    results = []
    start_index = 1
    while len(results) < num_results:
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'start': start_index
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        if 'items' not in data:
            break
        for item in data['items']:
            results.append({
                'title': item.get('title'),
                'link': item.get('link'),
                'snippet': item.get('snippet')
            })
            if len(results) >= num_results:
                break
        start_index += len(data['items'])

    SEARCH_CACHE[cache_key] = results
    return results


async def fetch_page_content(browser_semaphore, context, url):
    if url in BROWSER_CACHE:
        LOGGER.info(f'CACHED BROWSER {url}')
        result = BROWSER_CACHE[url]
        if result.strip():
            # guard against an empty page
            return result
        else:
            LOGGER.info('CACHED RESULT is empty, running anyway')
    LOGGER.info(f'NOT CACHED BROWSER {url}')

    domain = urlparse(url).netloc

    async with GLOBAL_SEMPAHORE:
        domain_semaphore = DOMAIN_SEMAPHORES[domain]

    if url.lower().endswith('.pdf'):
        # Handle PDF outside Playwright, by just downloading and parsing.
        async with domain_semaphore, aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    pdf_data = await response.read()
                    # Parse PDF
                    pdf_file = io.BytesIO(pdf_data)
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = []
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                    content = '\n'.join(text)
                else:
                    content = ''  # or handle error if needed

        BROWSER_CACHE[url] = content
        return content

    else:
        # Handle normal HTML pages with Playwright
        async with domain_semaphore, browser_semaphore:
            content = ''
            try:
                page = await context.new_page()
                await page.goto(url, timeout=10000)  # maybe a longer timeout for safety
                raw_content = await page.content()
                content = '\\n'.join([
                    x.strip() for x in extract_text_elements(raw_content)
                    if x.strip()])
            finally:
                await page.close()
                BROWSER_CACHE[url] = content

        return content


async def handle_question_species(
    row,
    species,
    openai_context,
    openai_semaphore,
    browser_semaphore,
    context
):
    column_prefix, question = [x.strip() for x in row.split(':')]
    species_question = question.format(species=species)

    search_result_list = await google_custom_search_async(
        API_KEY, CX, species_question, num_results=25
    )
    tasks = [
        asyncio.create_task(
            process_one_search_result(
                browser_semaphore,
                context,
                openai_semaphore,
                openai_context,
                question,
                species_question,
                sr,
                species
            ))
        for sr in search_result_list
    ]
    # (snippet, answer) tuples in answer list
    answer_list = await asyncio.gather(*tasks, return_exceptions=True)
    aggregate_answer = await aggregate_answers(openai_semaphore, openai_context, answer_list)

    return column_prefix, question, species, aggregate_answer, search_result_list, answer_list


def extract_text_elements(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        extracted_text_list = []
        for element in text_elements:
            local_text = element.get_text(strip=True)
            local_text = re.sub(r'\s+', ' ', local_text)
            local_text = re.sub(r'[^\x20-\x7E]+', '', local_text)
            extracted_text_list.append(local_text)
        return extracted_text_list
    except Exception as e:
        LOGGER.exception(f'MASSIVE ERROR {e}')


def cache_key(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


async def make_request_with_backoff(openai_semaphore, openai_context, chat_args, max_retries=5):
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            async with openai_semaphore:
                response = openai_context['client'].chat.completions.create(**chat_args)
            return response
        except Exception as e:
            LOGGER.exception(f'attempt {attempt} failed on openai exception {e} on {chat_args}')
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            if backoff > MAX_BACKOFF_WAIT:
                backoff = MAX_BACKOFF_WAIT


async def get_webpage_answers(openai_semaphore, openai_context, answer_context, query_template, full_query, source_url):
    messages = [
        {'role': 'developer',
         'content': 'You are given a snippet of text from a webpage that showed up in a Google search after querying the user question below. If you can find a specific succinct answer to that question return that answer with no other text. If there are multiple answers, return them all, comma separated. If no answer is found return UNKNOWN. Respond with no additional text besides the brief answer (s) or UNKNOWN.'},
        {'role': 'user',
         'content': full_query},
        {'role': 'assistant',
         'content': f'RELEVANT WEBPAGE SNIPPET: {answer_context}'}
    ]

    response_text = await generate_text(openai_semaphore, openai_context, 'gpt-4o-mini', messages, source_url)
    return response_text


async def generate_text(openai_semaphore, openai_context, model, messages, source_url=None):
    chat_args = {'model': model, 'messages': messages}
    key = cache_key(chat_args)
    if key in PROMPT_CACHE:
        LOGGER.info('CACHED openai')
        response_text = PROMPT_CACHE[key]
        LOGGER.info(f'source url: {source_url} -- got that CACHED response {response_text[:100]}')
    else:
        LOGGER.info('NOT CACHED openai')
        LOGGER.info(messages)
        response = await make_request_with_backoff(openai_semaphore, openai_context, chat_args)
        response_text = response.choices[0].message.content
        PROMPT_CACHE[key] = response_text
    return response_text


def validate_query_template(value):
    if '{subject}' not in value:
        raise argparse.ArgumentTypeError("The query template must contain '{subject}'")
    return value


async def aggregate_answers(openai_semaphore, openai_context, answer_list):
    combined_answers = '; '.join([answer[1] for answer in answer_list])
    messages = [
        {
            'role': 'developer',
            'content': (
                'You are given a list of answers from a question answering system. '
                'They may contain duplicates, partial overlaps, or the placeholder UNKNOWN. '
                'Combine them into a concise, readable response. '
                'Here is the required behavior: \n'
                '1. Remove duplicates.\n'
                '2. If there are any "UNKNOWN" entries, omit them unless every entry is "UNKNOWN".\n'
                '3. If after removing UNKNOWN entries there is nothing left, return "UNKNOWN".\n'
                '4. The final response can contain multiple words, but keep it succinct.\n'
                '5. Do not add extra commentary or explanation, just the concise result.'
            )
        },
        {
            'role': 'user',
            'content': 'Combine multiple answers into a simpler one.'
        },
        {
            'role': 'assistant',
            'content': f'Combined answers: {combined_answers}'
        }
    ]
    consolidated_answer = await generate_text(openai_semaphore, openai_context, 'gpt-4o-mini', messages)
    return consolidated_answer


async def process_one_search_result(
        browser_semaphore,
        context,
        openai_semaphore,
        openai_context,
        question,
        species_question,
        search_result,
        species):
    async with browser_semaphore:
        text_content = await fetch_page_content(
            browser_semaphore, context, search_result['link'])

    return text_content, await get_webpage_answers(openai_semaphore, openai_context, text_content, question, species_question, search_result['link'])


async def main():
    parser = argparse.ArgumentParser(description="Process a list of queries from a file.")
    parser.add_argument(
        'species_list_file', type=argparse.FileType('r'), help="Path to txt file listing all species")
    parser.add_argument(
        'question_list', type=argparse.FileType('r'), help="Path to the file containing query subjects of the form [header]: [question] on each line")
    parser.add_argument(
        '--max_subjects', type=int, help='limit to this many subjects for debugging reasons')
    parser.add_argument(
        '--headless_off', action='store_true')
    args = parser.parse_args()

    browser_semaphore = asyncio.Semaphore(MAX_TABS)
    openai_semaphore = asyncio.Semaphore(MAX_OPENAI)

    species_list = [
        line for line in args.species_list_file.read().splitlines()
        if line.strip()
    ]

    question_list = [
        line for line in args.question_list.read().splitlines()
        if line.strip()
    ]

    openai_context = create_openai_context()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=args.headless_off)
        context = await browser.new_context(
            accept_downloads=False,
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1280, 'height': 800},
            locale='en-US')

        question_answers = collections.defaultdict(list)
        tasks = []
        for row in question_list:
            for species in species_list:
                tasks.append(
                    asyncio.create_task(
                        handle_question_species(
                            row,
                            species,
                            openai_context,
                            openai_semaphore,
                            browser_semaphore,
                            context
                        )
                    )
                )

        # Run them all in parallel
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        await context.close()

    # Aggregate final results
    for column_prefix, question, species, aggregate_answer, search_result_list, answer_list in all_results:
        question_answers[f'{column_prefix} - {question}'].append(
            (question, species, search_result_list, aggregate_answer, answer_list)
        )

    LOGGER.info('all done with trait search pipeline, making table')
    main_data = {}
    details_list = []
    for col_prefix, qa_list in question_answers.items():
        for question, species, search_result_list, aggregate_answer, answer_list in qa_list:
            if species not in main_data:
                main_data[species] = {}
            main_data[species][question] = aggregate_answer
            for search_item, (snippet, answer_str) in zip(search_result_list, answer_list):
                details_list.append({
                    'species': species,
                    'question': question,
                    'answer': answer_str,
                    'url': search_item['link'],
                    'webpage title': search_item['title'],
                    'context': snippet
                })

    LOGGER.info('details all set, building tables now')
    species_set = sorted(list(main_data.keys()))
    questions_set = sorted(list({q for s in main_data for q in main_data[s].keys()}))
    rows = []
    for sp in species_set:
        row_vals = {}
        row_vals['species'] = sp
        for q in questions_set:
            row_vals[q] = main_data[sp].get(q, '')
        rows.append(row_vals)
    main_df = pd.DataFrame(rows)

    # Create detail DataFrame
    detail_df = pd.DataFrame(details_list)

    # Generate filenames with timestamps
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    main_filename = f'trait_search_results_{timestamp}.csv'
    detail_filename = f'trait_search_detailed_results_{timestamp}.csv'

    # Save to CSV
    LOGGER.info('about to save tables')
    main_df.to_csv(main_filename, index=False)
    detail_df.to_csv(detail_filename, index=False)
    LOGGER.info(f'tables saved to {main_filename} and {detail_filename}')


async def test():
    browser_semaphore = asyncio.Semaphore(MAX_TABS)
    openai_semaphore = asyncio.Semaphore(MAX_OPENAI)

    # url = 'https://vtfishandwildlife.com/sites/fishandwildlife/files/documents/About%20Us/Budget%20and%20Planning/WAP2015/5.-SGCN-Lists-Taxa-Summaries-%282015%29.pdf'
    # url = 'https://www.columbiatribune.com/story/lifestyle/family/2016/05/18/amazing-adaptations/21809648007/'
    url = 'https://cropwatch.unl.edu/2016/aphids-active-nebraska-spring-alfalfa/'
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(
            accept_downloads=False,
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1280, 'height': 800},
            locale='en-US')
        content = await fetch_page_content(browser_semaphore, context, url)
        openai_context = create_openai_context()
        question = 'What species does acyrthosiphon pisum eat?'
        answers = await get_webpage_answers(openai_semaphore, openai_context, content, question, question, url)
        # result = await fetch_relevant_snippet(browser_semaphore, context, url, question, 'acyrthosiphon pisum')

        print(f'result: {answers}')


if __name__ == '__main__':
    print('about to search')
    asyncio.run(main())
    # asyncio.run(test())
