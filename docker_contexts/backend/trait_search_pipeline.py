import argparse
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import requests
import spacy
from duckduckgo_search import DDGS


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
    query = args['query']
    detailed_snippet = fetch_relevant_snippet(result['href'], query)
    detailed_snippet = f'{detailed_snippet} {result["body"]}' if detailed_snippet else result['body']
    return {
        'body': f'{result["title"]}: {detailed_snippet}',
        'href': result['href']}


def parse_results(raw_results, query):
    detailed_snippet_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        detailed_snippet_list = list(executor.map(
            process_result,
            ({'result': result, 'query': query} for result in raw_results)))

    return detailed_snippet_list


def find_relevant_snippets_nlp(text_elements, query):
    query_doc = spacy_tokenizer(query)
    query_tokens = set([token.lemma_ for token in query_doc if not token.is_stop])
    relevant_snippets = []
    for text in text_elements:
        text_doc = spacy_tokenizer(text)
        text_tokens = set([token.lemma_ for token in text_doc if not token.is_stop])
        if query_tokens & text_tokens:
            relevant_snippets.append(text)
    return '\n'.join(relevant_snippets)


def fetch_page_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.content


def extract_text_elements(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    return [element.get_text(strip=True) for element in text_elements]


def fetch_relevant_snippet(url, query):
    try:
        html_content = fetch_page_content(url)
        text_elements = extract_text_elements(html_content)
        relevant_snippets = find_relevant_snippets_nlp(text_elements, query)
        return relevant_snippets
    except Exception as e:
        return ''


def process_snippet(args):
    snippet = args['snippet']
    query_template = args['query_template']
    full_query = args['full_query']
    result = qa_pipeline(question=full_query + ' Answer in one word.', context=snippet['body'])
    query_doc = spacy_tokenizer(query_template)
    query_tokens = set([token.lemma_ for token in query_doc if not token.is_stop])
    answer_doc = spacy_tokenizer(result['answer'])
    answer_tokens = set([token.lemma_ for token in answer_doc if not token.is_stop])

    if query_tokens & answer_tokens:
        return (result['score'], result['answer'], snippet['body'], snippet['href'])
    else:
        return (0.0, None, '', '')


def get_answers(detailed_snippet_list, query_template, full_query):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        answers = list(executor.map(
            process_snippet,
            ({'snippet': snippet, 'query_template': query_template, 'full_query': full_query}
             for snippet in detailed_snippet_list)))
    return answers


def process_subject(subject, args):
    subject = subject.strip()
    query = args.query_template.format(subject=subject)
    print(query)
    raw_results = duckduckgo_search(query)
    print(f'1/3 QUERY COMPLETE - {query}')
    detailed_snippet_list = parse_results(raw_results, query)
    print(f'2/3 SNIPPETS COMPLETE - {query}')
    answers = get_answers(detailed_snippet_list, args.query_template, query)
    print(f'3/3 ANSWERS COMPLETE - {query}')
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
        print(f'{subject} {score:.2f}: {answer}')
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
        result = process_subject(subject, args)
        if result is not None:
            df = pd.DataFrame([result])
            df.to_csv(query_result_filename, mode='a', index=False, header=False)
