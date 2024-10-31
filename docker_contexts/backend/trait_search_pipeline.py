import asyncio
import logging
import csv
import argparse
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from contextlib import contextmanager

from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import spacy
from duckduckgo_search import DDGS

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

from playwright.sync_api import sync_playwright

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
        'rquest',
        'duckduckgo_search.DDGS',
        ]:
    logging.getLogger(module).setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)


# @contextmanager
# def get_driver():
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")  # Run in headless mode
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--window-size=1920,1080")
#     chrome_options.add_argument("--disable-extensions")
#     chrome_options.add_argument("--proxy-server='direct://'")
#     chrome_options.add_argument("--proxy-bypass-list=*")
#     chrome_options.add_argument("--start-maximized")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_argument("--no-sandbox")
#     # Add other options as needed

#     driver = webdriver.Chrome(options=chrome_options)

#     headers = {
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
#         'Accept-Language': 'en-US,en;q=0.9',
#         'Accept-Encoding': 'gzip, deflate, br, zstd',
#         'dnt': '1',
#         'pragma': 'no-cache',
#         'priority': 'u=0, i',
#         'sec-ch-ua': '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
#         'sec-ch-ua-mobile': '?0',
#         'sec-ch-ua-platform': "Windows",
#         'sec-fetch-dest': 'document',
#         'sec-fetch-mode': 'navigate',
#         'sec-fetch-site': 'none',
#         'sec-fetch-user': '?1',
#         'sec-gpc': '1',
#         'upgrade-insecure-requests': '1',
#     }
#     driver.execute_cdp_cmd("Network.enable", {})
#     driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": headers})
#     driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     chrome_options.add_argument("--user-data-dir=./custom_profile_chrome")  # Persistent session storage
#     try:
#         yield driver
#     finally:
#         driver.quit()


def duckduckgo_search(query):
    results = DDGS().text(query, max_results=20)
    return results


spacy_tokenizer = spacy.load('en_core_web_sm')
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device='cuda')


def process_result(args):
    result = args['result']
    query_template = args['query_template']
    subject = args['subject']
    detailed_snippet = fetch_relevant_snippet(
        result['href'],
        query_template.format(subject=''),
        subject)
    detailed_snippet = f'{detailed_snippet} {result["body"]}' if detailed_snippet else result['body']
    return {
        'body': f'{result["title"]}: {detailed_snippet}',
        'href': result['href']}


def parse_search_results(raw_search_results, query_template, subject):
    detailed_snippet_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        detailed_snippet_list = list(executor.map(
            process_result,
            ({'result': result,
              'query_template': query_template,
              'subject': subject,
              'index': index,
              'n_to_process': len(raw_search_results)}
             for index, result in enumerate(raw_search_results))))
    return detailed_snippet_list


def find_relevant_snippets_nlp(text_elements, query_template, subject):
    query_tokens = set([token.lemma_ for token in spacy_tokenizer(query_template) if not token.is_stop])
    subject_tokens = set([token.lemma_ for token in spacy_tokenizer(subject) if not token.is_stop])
    relevant_snippets = []
    for text in text_elements:
        text_doc = spacy_tokenizer(text)
        text_tokens = set([token.lemma_ for token in text_doc if not token.is_stop])
        if (query_tokens & text_tokens) and (subject_tokens & text_tokens):
            relevant_snippets.append(text)
    return ' '.join(relevant_snippets)


def fetch_page_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        browser.close()
        return content


def extract_text_elements(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    return [element.get_text(strip=True) for element in text_elements]


def fetch_relevant_snippet(url, query_template, subject):
    try:
        html_content = fetch_page_content(url)
        text_elements = extract_text_elements(html_content)
        relevant_snippets = find_relevant_snippets_nlp(
            text_elements, query_template, subject)
        return relevant_snippets
    except Exception as e:
        LOGGER.exception(f'error on {url}')
        return ''


def answer_question_with_context(args):
    context = args['context']
    LOGGER.debug(f'processing context: {context}')
    query_template = args['query_template']
    full_query = args['full_query']
    result = qa_pipeline(
        question=full_query + ' Answer in one word.',
        context=context['body'])
    query_doc = spacy_tokenizer(query_template)
    query_tokens = set(
        [token.lemma_ for token in query_doc if not token.is_stop])
    answer_doc = spacy_tokenizer(result['answer'])
    answer_tokens = set(
        [token.lemma_ for token in answer_doc if not token.is_stop])

    if query_tokens & answer_tokens:
        return (result['score'], result['answer'], context['body'], context['href'])
    else:
        return (0.0, None, '', '')


def get_answers(cleaned_context_list, query_template, full_query):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        answers = list(executor.map(
            answer_question_with_context,
            ({'context': context,
              'query_template': query_template,
              'full_query': full_query}
             for context in cleaned_context_list)))
    return answers


def answer_question(subject, args):
    subject = subject.strip()
    query = args.query_template.format(subject=subject)
    raw_search_results = duckduckgo_search(query)
    LOGGER.info(f'1/3 INTERNET SEARCH COMPLETE - {query}')
    detailed_snippet_list = parse_search_results(raw_search_results, args.query_template, subject)
    LOGGER.info(f'2/3 CLEANED INTERNET SNIPPETS COMPLETE - {query}')
    answers = get_answers(detailed_snippet_list, args.query_template, query)
    LOGGER.info(f'3/3 ANSWERS COMPLETE - {query}')
    answer_to_score = collections.defaultdict(lambda: (0, ''))

    for score, answer, snippet, href in answers:
        if answer is None:
            continue
        answer = answer.lower().strip()
        current_tuple = answer_to_score[answer]
        answer_to_score[answer] = (
            current_tuple[0] + score, current_tuple[1] + f'({href}) - ' + snippet + '\n'
        )

    # Get the best answer with the highest score
    sorted_answers = sorted(answer_to_score.items(), key=lambda x: x[1][0], reverse=True)
    if sorted_answers:
        answer, (score, snippet) = sorted_answers[0]
        LOGGER.info(f'{subject} {score:.2f}: {answer}')
        return {
            'subject': subject,
            'answer': answer,
            'score': score,
            'snippet': snippet
        }
    else:
        return {
            'subject': subject,
            'answer': '-',
            'score': -1,
            'snippet': ''
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a list of queries from a file.")
    parser.add_argument(
        'query_template', help="Query and insert {subject} to be queried around")
    parser.add_argument(
        'query_subject_list', type=argparse.FileType('r'), help="Path to the file containing query subjects")
    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    query_result_filename = f'query_result_{current_time}.csv'

    # Initialize CSV with headers
    pd.DataFrame(columns=['subject', 'answer', 'score', 'snippet']).to_csv(query_result_filename, index=False)

    with args.query_subject_list as subject_list_file:
        subjects = subject_list_file.readlines()

    # Process subjects in parallel and stream results to CSV
    for subject in subjects:
        LOGGER.info(f'processing {subject}')
        answer = answer_question(subject, args)
        if answer is not None:
            LOGGER.info(f'answer: {answer["subject"]}:{answer["answer"]}')
            df = pd.DataFrame([answer])
            df.to_csv(
                query_result_filename,
                mode='a',
                index=False,
                header=False,
                quoting=csv.QUOTE_ALL)
        break
