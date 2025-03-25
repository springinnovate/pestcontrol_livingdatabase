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
import time
import warnings

from bs4 import BeautifulSoup
from diskcache import Cache
from openai import OpenAI
from playwright.async_api import async_playwright
from transformers import pipeline
import pandas as pd
import requests
import spacy

LOGGER = logging.getLogger(__name__)

SEARCH_CACHE = Cache('google_search_cache')
PROMPT_CACHE = Cache('openai_cache')
BROWSER_CACHE = Cache('browser_cache')
NLP_CACHE = Cache('nlp_cache')
# avoid searching the same domain at once
DOMAIN_SEMAPHORES = defaultdict(lambda: asyncio.Semaphore(1))

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


MAX_TABS = 50

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


spacy_tokenizer = spacy.load('en_core_web_sm')
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device='cuda')


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


async def fetch_page_content(browser_semaphore, browser, url):
    if url in BROWSER_CACHE:
        print(f'CACHED BROWSER {url}')
        result = BROWSER_CACHE[url]
        if result.strip():
            # guard against an empty page
            return result
    print(f'NOT CACHED BROWSER {url}')

    domain = urlparse(url).netloc

    domain_semaphore = DOMAIN_SEMAPHORES[domain]

    async with domain_semaphore:
        async with browser_semaphore:
            context = None
            content = ''
            try:
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(url, timeout=5000)
                content = await page.content()
                return content
            except Exception as e:
                return f'ERROR: could not load page because of {e}'
            finally:
                BROWSER_CACHE[url] = content
                if context is not None:
                    await context.close()


async def handle_question_species(
    row,
    species,
    openai_context,
    browser_semaphore,
    browser
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
                browser,
                openai_context,
                question,
                species_question,
                sr,
                species
            ))
        for sr in search_result_list
    ]
    # (snippet, answer) tuples in answer list
    answer_list = await asyncio.gather(*tasks)
    aggregate_answer = aggregate_answers(openai_context, answer_list)

    return column_prefix, question, species, aggregate_answer, search_result_list, answer_list


def filter_by_query_and_subject(query_tokens, subject_tokens, candidate_text):
    text_tokens = set([
        token.lemma_
        for token in spacy_tokenizer(candidate_text)
        if not token.is_stop and token.lemma_.strip() != ''])
    if (query_tokens & text_tokens) and (subject_tokens & text_tokens):
        return candidate_text
    return None


def find_relevant_snippets_nlp(text_elements, question, subject):
    cache_key = json.dumps({
        'text_elements': text_elements,
        'question': question,
        'subject': subject
    }, sort_keys=True)

    if cache_key in NLP_CACHE:
        return NLP_CACHE[cache_key]
    try:
        query_tokens = set(
            [token.lemma_
             for token in spacy_tokenizer(question)
             if not token.is_stop and token.lemma_.strip() != ''])
        subject_tokens = set(
            [token.lemma_
             for token in spacy_tokenizer(subject)
             if not token.is_stop and token.lemma_.strip() != ''])

        task_list = [
            filter_by_query_and_subject(query_tokens, subject_tokens, candidate_text)
            for candidate_text in text_elements
        ]
        result_str = ' '.join([
            filtered_text for filtered_text in task_list
            if filtered_text is not None])
        NLP_CACHE[cache_key] = result_str
        return result_str
    except Exception:
        LOGGER.exception('bad stuff')


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
        print(f'MASSIVE ERROR {e}')


async def fetch_relevant_snippet(browser_semaphore, browser, url, question, subject):
    try:
        html_content = await fetch_page_content(browser_semaphore, browser, url)
        text_elements = extract_text_elements(html_content)
        relevant_snippets = find_relevant_snippets_nlp(
            text_elements, question, subject)
        return relevant_snippets
    except Exception as e:
        return 'got exception: ' + str(e)


def normalize_answer(answer):
    tokens = spacy_tokenizer(answer.lower())
    return " ".join([token.lemma_ for token in tokens if not token.is_stop])


def split_contexts_into_chunks(context, max_length, tokenizer, question):
    chunks = []
    question_length = len(encode_text(question, tokenizer))
    max_context_length = max_length - question_length - 3
    context_tokens = encode_text(context, tokenizer)
    context_length = len(context_tokens)
    if context_length <= max_context_length:
        chunks.append(context)
    else:
        # Context is too long, need to split it into smaller chunks
        for i in range(0, context_length, max_context_length):
            chunk_tokens = context_tokens[i:i + max_context_length]
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            chunks.append(chunk_text)
    return chunks


def cache_key(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


MAX_BACKOFF = 10


def make_request_with_backoff(openai_context, chat_args, max_retries=20):
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = openai_context['client'].chat.completions.create(**chat_args)
            return response
        except Exception as e:
            LOGGER.warn(f'attempt {attempt} failed on openai exception {e} on {chat_args}')
            if attempt == max_retries:
                raise
            time.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            if backoff > MAX_BACKOFF:
                backoff = MAX_BACKOFF


def get_webpage_answers(openai_context, answer_context, query_template, full_query):
    messages = [
        {'role': 'developer',
         'content': 'You are given a snippet of text from a webpage that showed up in a Google search after querying the user question below. If you can find a specific succinct answer to that question return that answer with no other text. If there are multiple answers, return them all, comma separated. If no answer is found return UNKNOWN. Respond with no additional text besides the brief answer (s) or UNKNOWN.'},
        {'role': 'user',
         'content': full_query},
        {'role': 'assistant',
         'content': f'RELEVANT WEBPAGE SNIPPET: {answer_context}'}
    ]

    response_text = generate_text(openai_context, 'gpt-4o-mini', messages)
    return response_text


def generate_text(openai_context, model, messages):
    chat_args = {'model': model, 'messages': messages}
    key = cache_key(chat_args)
    if key in PROMPT_CACHE:
        print('CACHED openai')
        response_text = PROMPT_CACHE[key]
    else:
        print('NOT CACHED openai')
        print(messages)
        response = make_request_with_backoff(openai_context, chat_args)
        response_text = response.choices[0].message.content
        PROMPT_CACHE[key] = response_text
    return response_text


def validate_query_template(value):
    if '{subject}' not in value:
        raise argparse.ArgumentTypeError("The query template must contain '{subject}'")
    return value


def aggregate_answers(openai_context, answer_list):
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
    consolidated_answer = generate_text(openai_context, 'gpt-4o-mini', messages)
    return consolidated_answer


async def process_one_search_result(
        browser_semaphore,
        browser,
        openai_context,
        question,
        species_question,
        search_result,
        species):
    async with browser_semaphore:
        snippet = await fetch_relevant_snippet(
            browser_semaphore,
            browser,
            search_result['link'],
            species_question,
            species
        )
    return snippet, get_webpage_answers(openai_context, snippet, question, species_question)


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
        browser = await playwright.chromium.launch(headless=not args.headless_off)

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
                            browser_semaphore,
                            browser
                        )
                    )
                )
            break

        # Run them all in parallel
        all_results = await asyncio.gather(*tasks)

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
                    'context (up to 5000 chars)': snippet[:5000]
                })

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
    main_df.to_csv(main_filename, index=False)
    detail_df.to_csv(detail_filename, index=False)
    LOGGER.info(f'tables saved to {main_filename} and {detail_filename}')

if __name__ == '__main__':
    print('about to search')
    asyncio.run(main())
