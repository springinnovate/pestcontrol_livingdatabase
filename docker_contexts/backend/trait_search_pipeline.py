from datetime import datetime
from functools import partial
import argparse
import asyncio
import collections
import csv
import logging
import re
import sys
import warnings

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from transformers import pipeline
import pandas as pd
import spacy
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

LOGGER = logging.getLogger(__name__)

SEARCH_ENGINE_ID = 'd7b8ff21e69824ba5'

try:
    API_KEY = open('./secrets/custom_search_api_key.txt', 'r').read()
except FileNotFoundError:
    LOGGER.exception(
        'custom_search_api_key.txt API key not found, should be in '
        'subidrectory of "secrets"')


def encode_text(text, tokenizer, max_length=512):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return tokens


MAX_TABS = 5

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
        'request']:
    logging.getLogger(module).setLevel(logging.ERROR)

spacy_tokenizer = spacy.load('en_core_web_sm')
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device='cuda')


async def searchengine_search(ddg_semaphore, query):
    async with ddg_semaphore:
        results = []
        while True:
            url = f'https://html.duckduckgo.com/html/?q={query.replace(" ", "+")}'
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Locate the main results container
                results_container = soup.find('div', id='links')
                if results_container:
                    # Find each result within the container
                    local_results = results_container.find_all('div', class_='links_main links_deep result__body')
                    for result in local_results:
                        # Extract the title and URL from <a> inside "result_title"
                        title_tag = result.find('a', class_='result__a')
                        result_title = title_tag.get_text(strip=True) if title_tag else 'No title'
                        result_link = title_tag['href'] if title_tag else 'No link'

                        # Extract the snippet text from <a> inside "result__snippet"
                        snippet_tag = result.find('a', class_='result__snippet')
                        result_snippet = snippet_tag.get_text(strip=True) if snippet_tag else 'No snippet'

                        # Print the extracted details
                        print("-" * 40)
                        results.append({
                            'title': result_title,
                            'link': result_link,
                            'snippet': result_snippet,
                        })
                else:
                    print("No results container found.")
            else:
                print(f"Failed to retrieve results. Status code: {response.status_code}")

            # url = "https://www.googleapis.com/customsearch/v1"
            # params = {
            #     "key": API_KEY,
            #     "cx": SEARCH_ENGINE_ID,
            #     "q": query
            # }

            # print(f'searching {params}')

            # response = requests.get(url, params=params)
            # loop = asyncio.get_event_loop()
            # try:
            #     results = await loop.run_in_executor(
            #         None,
            #         partial(DDGS().text, query, max_results=20))
            # except duckduckgo_search.exceptions.RatelimitException:
            #     LOGGER.warning('duckduckgo rate limit error, sleeping for a bit')
            #     await asyncio.sleep(3)
            #     continue
            # return results


async def fetch_page_content(browser_semaphore, browser, url):
    async with browser_semaphore:
        context = None
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=5000)
            content = await page.content()
            return content
        finally:
            if context is not None:
                await context.close()


async def extract_query_relevant_text(args):
    result = args['result']
    # query_template = args['query_template']
    # subject = args['subject']
    # detailed_snippet = result['snippet_text']

    # await fetch_relevant_snippet(
    #     args['browser_semaphore'],
    #     args['browser'],
    #     result['href'],
    #     query_template.format(subject=''),
    #     subject)
    # detailed_snippet = f'{detailed_snippet} {result["body"]}' if detailed_snippet else result['body']
    return {
        'body': f'{result["title"]}: {result["snippet"]}',
        'href': result['link']}


async def extract_relevant_text_from_search_results(
        browser_semaphore, browser, raw_search_results, query_template, subject):
    tasks = []
    for index, result in enumerate(raw_search_results):
        tasks.append(extract_query_relevant_text({
            'result': result,
            'browser_semaphore': browser_semaphore,
            'browser': browser,
            'query_template': query_template,
            'subject': subject,
            'index': index,
            'n_to_process': len(raw_search_results)
        }))
    relevant_text_list = await asyncio.gather(*tasks)
    return relevant_text_list


async def filter_by_query_and_subject(query_tokens, subject_tokens, candidate_text):
    text_tokens = set([
        token.lemma_
        for token in spacy_tokenizer(candidate_text)
        if not token.is_stop and token.lemma_.strip() != ''])
    if (query_tokens & text_tokens) and (subject_tokens & text_tokens):
        return candidate_text
    return None


async def find_relevant_snippets_nlp(text_elements, query_template, subject):
    try:
        query_tokens = set(
            [token.lemma_
             for token in spacy_tokenizer(query_template)
             if not token.is_stop and token.lemma_.strip() != ''])
        subject_tokens = set(
            [token.lemma_
             for token in spacy_tokenizer(subject)
             if not token.is_stop and token.lemma_.strip() != ''])

        task_list = [
            filter_by_query_and_subject(query_tokens, subject_tokens, candidate_text)
            for candidate_text in text_elements
        ]
        return ' '.join([
            filtered_text for filtered_text in await asyncio.gather(*task_list)
            if filtered_text is not None])
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


async def fetch_relevant_snippet(browser_semaphore, browser, url, query_template, subject):
    try:
        html_content = await fetch_page_content(browser_semaphore, browser, url)
        text_elements = extract_text_elements(html_content)
        relevant_snippets = await find_relevant_snippets_nlp(
            text_elements, query_template, subject)
        return relevant_snippets
    except Exception as e:
        return 'got exception: ' + str(e)


def normalize_answer(answer):
    tokens = spacy_tokenizer(answer.lower())
    return " ".join([token.lemma_ for token in tokens if not token.is_stop])


def split_contexts_into_chunks(contexts, max_length, tokenizer, question):
    chunks = []
    question_length = len(encode_text(question, tokenizer))
    max_context_length = max_length - question_length - 3
    for context, href in contexts:
        context_tokens = encode_text(context, tokenizer)
        context_length = len(context_tokens)
        if context_length <= max_context_length:
            chunks.append((context, href))
        else:
            # Context is too long, need to split it into smaller chunks
            for i in range(0, context_length, max_context_length):
                chunk_tokens = context_tokens[i:i + max_context_length]
                chunk_text = tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                chunks.append((chunk_text, href))
    return chunks


async def get_answers(cleaned_context_list, query_template, full_query, context):
    try:
        max_length = qa_pipeline.tokenizer.model_max_length  # Typically 512 for BERT models
        contexts = [(context['body'], context['href']) for context in cleaned_context_list]
        chunks = split_contexts_into_chunks(
            contexts, max_length, qa_pipeline.tokenizer, full_query + f' {context}.')
        answers = []
        for chunk, href in chunks:
            result = qa_pipeline(
                question=full_query + f' {context}',
                context=chunk)
            query_tokens = set(
                token.lemma_ for token in spacy_tokenizer(query_template) if not token.is_stop
            )

            answer_tokens = set(
                token.lemma_
                for token in spacy_tokenizer(result['answer'])
                if not token.is_stop
            )

            # Check if the answer is relevant
            if query_tokens & answer_tokens:
                answers.append(
                    (result['score'], normalize_answer(result['answer']), chunk, href)
                )
        return answers
    except Exception:
        LOGGER.exception('somethinb bad happened answering questions')
        raise


async def answer_question(ddg_semaphore, browser_semaphore, browser, subject, args):
    try:
        subject = subject.strip()
        query = args.query_template.format(subject=subject)
        LOGGER.debug(query)
        sys.exit()
        raw_search_results = await searchengine_search(ddg_semaphore, query)
        LOGGER.info(f'1/3 INITAL INTERNET SEARCH COMPLETE - {query}')
        relevant_text_list = await extract_relevant_text_from_search_results(
            browser_semaphore, browser, raw_search_results, args.query_template, subject)
        LOGGER.info(f'2/3 RELEVANT TEXT EXTRACTED FROM SEARCH - {query}')
        answers = await get_answers(
            relevant_text_list, args.query_template, query, args.context)
        LOGGER.info(f'3/3 ANSWERS ARE ANSWERED - {query}')
        answer_to_score = collections.defaultdict(lambda: (0, '', 0))

        base_urls = '\n'.join([
            search_result['href'] for search_result in raw_search_results])

        for score, answer, snippet, href in answers:
            print(f'ANSWER: {answer} -- SNIPPET: {snippet}'.encode('utf-8', errors='replace').decode('utf-8'))
            if answer in [None, '']:
                continue
            answer = answer.lower().strip()
            current_tuple = answer_to_score[answer]
            answer_to_score[answer] = (
                current_tuple[0] + score,
                current_tuple[1] + f'*************\n({href}) - ' + snippet + '\n',
                current_tuple[2] + 1
            )

        # Get the best answer with the highest score
        sorted_answers = sorted(answer_to_score.items(), key=lambda x: x[1][0], reverse=True)
        if sorted_answers:
            answer, (score, snippet, consensus_answer_count) = sorted_answers[0]
            all_answers = '\n\t'.join([
                f'{score:.2f}:{answer}'
                for score, answer, _f, _ in
                sorted(answers, key=lambda x: x[0], reverse=True)])
            LOGGER.debug('ALL SCORES:\n\t' + all_answers)
            LOGGER.info(f'*** FINAL ANSWER {subject} {score:.2f}: {answer}')
            return {
                'subject': subject,
                'answer': answer,
                'confidence score': score,
                'consensus answer count': f'{consensus_answer_count} of {len(answers)}',
                'text used to get answer': snippet,
                'all answers': all_answers,
                'urls searched': base_urls
            }
        else:
            return {
                'subject': subject,
                'answer': '-',
                'confidence score': -1,
                'consensus answer count': 0,
                'text used to get answer': '',
                'all answers': '',
                'urls searched': base_urls
            }
        LOGGER.info(f'4/4 TOP ANSWER IS SAVED - {query}')
    except Exception as e:
        LOGGER.exception('something bad happened')
        raise


def validate_query_template(value):
    if '{subject}' not in value:
        raise argparse.ArgumentTypeError("The query template must contain '{subject}'")
    return value


async def main():
    parser = argparse.ArgumentParser(description="Process a list of queries from a file.")
    parser.add_argument(
        'query_template',
        type=validate_query_template,
        help="Query and insert {subject} to be queried around")
    parser.add_argument(
        'query_subject_list', type=argparse.FileType('r'), help="Path to the file containing query subjects")
    parser.add_argument(
        '--max_subjects', type=int, help='limit to this many subjects for debugging reasons')
    parser.add_argument(
        '--context', default='',
        help='Additional text info to include when asking a question that is not part of the raw subject.')
    parser.add_argument(
        '--headless_off', action='store_true')
    args = parser.parse_args()

    ddg_semaphore = asyncio.Semaphore(1)
    browser_semaphore = asyncio.Semaphore(MAX_TABS)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headless_off)

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        query_result_filename = f'query_result_{current_time}.csv'

        # Initialize CSV with headers
        pd.DataFrame(columns=['subject', 'answer', 'score', 'consensus answer count', 'text used to get answer', 'all answers', 'urls searched']).to_csv(query_result_filename, index=False)

        with args.query_subject_list as subject_list_file:
            subjects = subject_list_file.readlines()

        # Process subjects in parallel and stream results to CSV
        tasks = []
        for index, subject in enumerate(subjects):
            subject = subject.strip()
            LOGGER.info(f'processing {subject}')
            tasks.append(answer_question(
                ddg_semaphore, browser_semaphore, browser, subject, args))
            if args.max_subjects is not None and index == args.max_subjects - 1:
                break
        for answer in await asyncio.gather(*tasks):
            if answer is not None:
                LOGGER.info(f'answer: {answer["subject"]}:{answer["answer"]}')
                df = pd.DataFrame([answer])
                df.to_csv(
                    query_result_filename,
                    mode='a',
                    index=False,
                    header=False,
                    quoting=csv.QUOTE_ALL)
    LOGGER.info(f'all done, results in: {query_result_filename}')


def test_search():
    # url = "https://www.googleapis.com/customsearch/v1"
    # params = {
    #     "key": API_KEY,
    #     "cx": 'd7b8ff21e69824ba5',
    #     "q": "is {subject} monophagous or polyphagous?"
    # }

    # print(f'searching {params}')
    # response = requests.get(url, params=params)
    # print(f'response {response}')
    # results = response.json().get("items", [])
    # print(f'results {results}')
    # for item in results:
    #     print(item["title"], item["link"])

    #https://html.duckduckgo.com/html/?q=is abax ater monophagous or polyphagous
    query = f'is abax ater monophagous or polyphagous'
    url = f'https://html.duckduckgo.com/html/?q={query.replace(" ", "+")}'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the main results container
        results_container = soup.find('div', id='links')
        if results_container:
            # Find each result within the container
            results = results_container.find_all('div', class_='links_main links_deep result__body')
            for result in results:
                # Extract the title and URL from <a> inside "result_title"
                title_tag = result.find('a', class_='result__a')
                title_text = title_tag.get_text(strip=True) if title_tag else 'No title'
                title_link = title_tag['href'] if title_tag else 'No link'

                # Extract the snippet text from <a> inside "result__snippet"
                snippet_tag = result.find('a', class_='result__snippet')
                snippet_text = snippet_tag.get_text(strip=True) if snippet_tag else 'No snippet'

                # Print the extracted details
                print(f"Title: {title_text}")
                print(f"Link: {title_link}")
                print(f"Snippet: {snippet_text}")
                print("-" * 40)
        else:
            print("No results container found.")
    else:
        print(f"Failed to retrieve results. Status code: {response.status_code}")


if __name__ == '__main__':
    print('about to search')
    #test_search()

    asyncio.run(main())
