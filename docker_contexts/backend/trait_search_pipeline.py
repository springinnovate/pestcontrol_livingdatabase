import re
import asyncio
import logging
import csv
import argparse
import collections
from functools import partial
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import spacy
from duckduckgo_search import DDGS
import duckduckgo_search.exceptions
from playwright.async_api import async_playwright

N_WORKERS = 1#os.cpu_count()

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
        'rquest',
        'duckduckgo_search.DDGS',
        ]:
    logging.getLogger(module).setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)

spacy_tokenizer = spacy.load('en_core_web_sm')
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device='cuda')

MAX_TABS = 10


async def searchengine_search(ddg_semaphore, query):
    async with ddg_semaphore:
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                partial(DDGS().text, query, max_results=20))
        except duckduckgo_search.exceptions.RatelimitException:
            LOGGER.warn('rate limit error, sleeping for a bit')
            await asyncio.sleep(3)
        return results


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
    query_template = args['query_template']
    subject = args['subject']
    detailed_snippet = await fetch_relevant_snippet(
        args['browser_semaphore'],
        args['browser'],
        result['href'],
        query_template.format(subject=''),
        subject)
    detailed_snippet = f'{detailed_snippet} {result["body"]}' if detailed_snippet else result['body']
    return {
        'body': f'{result["title"]}: {detailed_snippet}',
        'href': result['href']}


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


def find_relevant_snippets_nlp(text_elements, query_template, subject):
    query_tokens = set(
        [token.lemma_
         for token in spacy_tokenizer(query_template)
         if not token.is_stop])
    subject_tokens = set(
        [token.lemma_
         for token in spacy_tokenizer(subject)
         if not token.is_stop])
    relevant_snippets = []
    for text in text_elements:
        text_doc = spacy_tokenizer(text)
        text_tokens = set([token.lemma_ for token in text_doc if not token.is_stop])
        # must be something about the query AND the subject in the text in
        # order to be considered
        if (query_tokens & text_tokens) and (subject_tokens & text_tokens):
            relevant_snippets.append(text)
    return ' '.join(relevant_snippets)


def extract_text_elements(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    extracted_text = [element.get_text(strip=True) for element in text_elements]
    combined_text = ' '.join(extracted_text)
    cleaned_text = re.sub(r'\s+', ' ', combined_text)
    cleaned_text = re.sub(r'[^\x20-\x7E]+', '', cleaned_text)
    return cleaned_text


async def fetch_relevant_snippet(browser_semaphore, browser, url, query_template, subject):
    try:
        html_content = await fetch_page_content(browser_semaphore, browser, url)
        text_elements = extract_text_elements(html_content)
        relevant_snippets = find_relevant_snippets_nlp(
            text_elements, query_template, subject)
        return relevant_snippets
    except Exception as e:
        return 'got exception: ' + str(e)


def normalize_answer(answer):
    tokens = spacy_tokenizer(answer.lower())
    return " ".join([token.lemma_ for token in tokens if not token.is_stop])


def answer_question_with_context(args):
    context = args['context']
    query_template = args['query_template']
    full_query = args['full_query']
    result = qa_pipeline(
        question=full_query + ' Answer in one word.',
        context=context['body'])
    query_tokens = set(
        [token.lemma_
         for token in spacy_tokenizer(query_template)
         if not token.is_stop])

    answer_tokens = set(
        [token.lemma_
         for token in spacy_tokenizer(result['answer'])
         if not token.is_stop])

    # make sure something from the query is part of the answer
    if query_tokens & answer_tokens:

        return (result['score'], normalize_answer(result['answer']), context['body'], context['href'])
    else:
        return (0.0, None, '', '')


def split_contexts_into_chunks(contexts, max_length, tokenizer, question):
    chunks = []
    current_chunk = ""
    current_length = 0
    question_length = len(tokenizer.encode(question, add_special_tokens=False))

    for context, href in contexts:
        context_length = len(tokenizer.encode(context, add_special_tokens=False))
        total_length = current_length + context_length + question_length + 3  # +3 for special tokens

        if total_length <= max_length:
            if current_chunk:
                current_chunk += " " + context
            else:
                current_chunk = context
            current_length += context_length
        else:
            if current_chunk:
                chunks.append((current_chunk, href))
            current_chunk = context
            current_length = context_length

    if current_chunk:
        chunks.append((current_chunk, href))

    return chunks


async def get_answers(cleaned_context_list, query_template, full_query):
    max_length = qa_pipeline.tokenizer.model_max_length  # Typically 512 for BERT models

    contexts = [(context['body'], context['href']) for context in cleaned_context_list]
    chunks = split_contexts_into_chunks(
        contexts, max_length, qa_pipeline.tokenizer, full_query + ' Answer in one word.')
    answers = []
    for chunk, href in chunks:
        print(href)
        result = qa_pipeline(
            question=full_query + ' Answer in one word.',
            context=chunk)

        query_tokens = set(
            token.lemma_ for token in spacy_tokenizer(query_template) if not token.is_stop
        )

        answer_tokens = set(
            token.lemma_ for token in spacy_tokenizer(result['answer']) if not token.is_stop
        )

        # Check if the answer is relevant
        if query_tokens & answer_tokens:
            answers.append(
                (result['score'], normalize_answer(result['answer']), chunk, href)
            )
    return answers


async def answer_question(ddg_semaphore, browser_semaphore, browser, subject, args):
    subject = subject.strip()
    query = args.query_template.format(subject=subject)
    raw_search_results = await searchengine_search(ddg_semaphore, query)
    LOGGER.info(f'1/4 INTERNET SEARCH COMPLETE - {query}')
    relevant_text_list = await extract_relevant_text_from_search_results(
        browser_semaphore, browser, raw_search_results, args.query_template, subject)
    LOGGER.info(f'2/4 RELEVANT TEXT EXTRACTED - {query}')
    answers = await get_answers(
        relevant_text_list, args.query_template, query)
    LOGGER.info(f'3/4 ANSWERS ARE ANSWERED - {query}')
    answer_to_score = collections.defaultdict(lambda: (0, ''))

    for score, answer, snippet, href in answers:
        if answer in [None, '']:
            continue
        answer = answer.lower().strip()
        LOGGER.debug(f'{href} -- {answer} -- {snippet}')
        current_tuple = answer_to_score[answer]
        answer_to_score[answer] = (
            current_tuple[0] + score, current_tuple[1] + f'*************\n({href}) - ' + snippet + '\n'
        )

    # Get the best answer with the highest score
    sorted_answers = sorted(answer_to_score.items(), key=lambda x: x[1][0], reverse=True)
    if sorted_answers:
        answer, (score, snippet) = sorted_answers[0]
        LOGGER.debug(
            'ALL SCORES:\n\t' +
            '\n\t'.join([f'{score:.2f}:{answer}'
                         for score, answer, _, _ in
                         sorted(answers, key=lambda x: x[0], reverse=True)]))
        LOGGER.info(f'*** FINAL ANSWER {subject} {score:.2f}: {answer}')
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
    LOGGER.info(f'4/4 TOP ANSWER IS SAVED - {query}')


async def main():
    parser = argparse.ArgumentParser(description="Process a list of queries from a file.")
    parser.add_argument(
        'query_template', help="Query and insert {subject} to be queried around")
    parser.add_argument(
        'query_subject_list', type=argparse.FileType('r'), help="Path to the file containing query subjects")
    args = parser.parse_args()

    ddg_semaphore = asyncio.Semaphore(1)
    browser_semaphore = asyncio.Semaphore(MAX_TABS)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        query_result_filename = f'query_result_{current_time}.csv'

        # Initialize CSV with headers
        pd.DataFrame(columns=['subject', 'answer', 'score', 'snippet']).to_csv(query_result_filename, index=False)

        with args.query_subject_list as subject_list_file:
            subjects = subject_list_file.readlines()

        # Process subjects in parallel and stream results to CSV
        tasks = []
        for index, subject in enumerate(subjects):
            LOGGER.info(f'processing {subject}')
            tasks.append(answer_question(ddg_semaphore, browser_semaphore, browser, subject, args))
            if index == 0:
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


if __name__ == '__main__':
    asyncio.run(main())
