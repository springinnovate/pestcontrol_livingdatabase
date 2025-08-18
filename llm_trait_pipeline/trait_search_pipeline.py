"""Pipeline to google search, follow links, then LLM questions.

DataforSEO for google search?
Then the webcrawler for webcrawling?


"""

from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse
import argparse
import asyncio
import hashlib
import httpx
import io
import json
import logging
import random
import re
import string
import sys
import time
import traceback
import warnings

from bs4 import BeautifulSoup
from diskcache import Cache, Index
from docx import Document
from itertools import islice
from openai import OpenAI
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from tqdm import tqdm
import aiohttp
import pandas as pd
import playwright
import PyPDF2
import requests
import tiktoken

LOGGER = logging.getLogger(__name__)
HTTP_TIMEOUT = 360.0
MAX_TABS = 25
MAX_BACKOFF_WAIT = 8
GOOGLE_QUERIES_PER_SECOND = 20

SEARCH_CACHE = Cache("google_search_cache")
PROMPT_CACHE = Index("openai_cache")
BROWSER_CACHE = Index("browser_cache")
PROCESS_URL_CACHE = Index("process_url_cache_v3")
SCRUBBED_OPENAI_CACHE = Index("scrubbed_by_4o_mini_v2")

GPT4O_MINI_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")


class SemaphoreWithTimeout:
    def __init__(
        self, value: int, timeout: float = None, name: str = "Default"
    ):
        self._semaphore = asyncio.Semaphore(value)
        self._timeout = timeout
        self._name = name

    async def __aenter__(self):
        try:
            start = time.time()
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._timeout
            )
            return self
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Semaphore acquire timed out after {time.time()-start:.2f}s!"
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        self._semaphore.release()


class DomainCooldown:
    def __init__(self, domain, cooldown_seconds=5):
        self.cooldown_seconds = cooldown_seconds
        # Initalize last access time to be longer than expected cooldown
        # so we start right away
        self.last_access = time.time() - cooldown_seconds - 1
        self.lock = asyncio.Lock()
        self.domain = domain
        self._domain_count = 0
        self._active_workers = 0

    async def __aenter__(self):
        self._active_workers += 1
        async with self.lock:
            self._domain_count += 1
            time_elapsed_since_last_access = time.time() - self.last_access
            time_to_wait = (
                self.cooldown_seconds - time_elapsed_since_last_access
            )
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
                self._active_workers -= 1
            self.last_access = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        # if self._active_workers > 0:
        #     LOGGER.debug(
        #         f"exiting semaphore for {self.domain}, {self._active_workers} "
        #         f"workers waiting"
        #     )


class DomainSemaphores(defaultdict):
    def __missing__(self, key):
        self[key] = DomainCooldown(key, 5.0)
        return self[key]


# avoid searching the same domain at once
DOMAIN_SEMAPHORES = DomainSemaphores()


try:
    API_KEY = open("./secrets/custom_search_api_key.txt", "r").read()
    CX = open("./secrets/cxid.txt", "r").read()
    OPENAI_KEY = open(
        "../../../LLMTechnicalWriter/.secrets/llm_grant_assistant_openai.key",
        "r",
    ).read()
except FileNotFoundError:
    LOGGER.exception(
        "custom_search_api_key.txt API key not found, should be in "
        'subidrectory of "secrets"'
    )


def encode_text(text, tokenizer, max_length=512):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tokens = tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        )
        return tokens


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)

for module in [
    "asyncio",
    "taskgraph",
    "selenium",
    "urllib3.connectionpool",
    "primp",
    "request",
    "httpcore",
    "openai",
    "httpx",
]:
    logging.getLogger(module).setLevel(logging.ERROR)


def create_openai_context():
    openai_client = OpenAI(api_key=OPENAI_KEY)
    openai_context = {
        "client": openai_client,
    }
    return openai_context


async def google_custom_search_async(
    rate_limit_semaphore, api_key, cx, query, num_results=25
):
    """
    Call-through to - google_custom_search_sync that otherwise couldn't be
    called async because the core requests library is not async compatable.
    """
    async with rate_limit_semaphore:
        try:
            result_was_cached = False
            (
                payload,
                result_was_cached,
            ) = await asyncio.to_thread(
                google_custom_search_sync,
                api_key,
                cx,
                query,
                num_results,
            )
            return payload
        finally:
            # wait 1 second before the next query
            if not result_was_cached:
                await asyncio.sleep(1)


def google_custom_search_sync(api_key, cx, query, num_results=25):
    cache_key = json.dumps(
        {
            "api_key": api_key,
            "cx": cx,
            "query": query,
            "num_results": num_results,
        },
        sort_keys=True,
    )

    if cache_key in SEARCH_CACHE:  # and (payload := SEARCH_CACHE[cache_key]):
        return SEARCH_CACHE[cache_key], True

    url = "https://www.googleapis.com/customsearch/v1"
    results = []
    start_index = 1
    while len(results) < num_results:
        params = {"key": api_key, "cx": cx, "q": query, "start": start_index}
        resp = requests.get(url, params=params)
        data = resp.json()
        if "items" not in data:
            break
        for item in data["items"]:
            results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
            )
            if len(results) >= num_results:
                break
        start_index += len(data["items"])

    SEARCH_CACHE[cache_key] = results
    return results, False


async def _download_pdf(session, url):
    async with session.get(url, allow_redirects=True) as response:
        if response.status != 200:
            return ""
        pdf_data = await response.read()
        pdf_file = io.BytesIO(pdf_data)
        reader = PyPDF2.PdfReader(pdf_file)
        text_pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
        return "\n".join(text_pages)


async def is_pdf(session, url, timeout_value):
    try:
        # try to read the head, that might work and if it does it's fastest
        async with session.head(url, allow_redirects=True) as head_response:
            if head_response.status == 200:
                content_type = head_response.headers.get(
                    "Content-Type", ""
                ).lower()
                if "pdf" in content_type:
                    return True
    except Exception as e:
        LOGGER.debug(f"pdf HEAD request failed for {url}: {e}")

    try:
        # Fallback on GET to try to read the header
        async with session.get(url, allow_redirects=True) as get_response:
            if get_response.status == 200:
                content_type = get_response.headers.get(
                    "Content-Type", ""
                ).lower()
                if "pdf" in content_type:
                    return True
    except Exception as e:
        LOGGER.debug(f"pdf GET request failed for {url}: {e}")

    return False


def extract_text_from_pdf(pdf_path):
    # with open(pdf_file, "rb") as file:
    #     pdf_data = io.BytesIO(file.read())
    # pdf_file = io.BytesIO(pdf_data)
    reader = PyPDF2.PdfReader(pdf_path)
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return "\n".join(text_pages)


def extract_text_from_docx(doc_path):
    doc = Document(doc_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def extract_text_from_doc(doc_path):
    def clean_extracted_text(text):
        # Keep printable characters plus newline, carriage return, and tab.
        LOGGER.debug(f"cleaning {len(text)} characters")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    import textract

    text = textract.process(doc_path)
    return clean_extracted_text(text.decode("utf-8"))


def filter_valid_words(phrase):
    translator = str.maketrans("", "", string.punctuation)
    cleaned_phrase = phrase.translate(translator)
    for word in cleaned_phrase.split():
        if bool(re.fullmatch(r"[A-Za-z]+", word)):
            yield word


def try_raw_text(doc_path):
    def clean_extracted_text(text):
        # Keep printable characters plus newline, carriage return, and tab.
        LOGGER.debug(f"cleaning {len(text)} characters")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    with open(doc_path, "r", encoding="utf-8", errors="ignore") as file:
        return " ".join(filter_valid_words(clean_extracted_text(file.read())))


def extract_text_from_binary_file(file_path):
    """
    Try to extract text from file_path by trying a series of extraction methods.
    If one fails, log the exception and try the next.
    """
    methods = [
        ("doc", extract_text_from_doc),
        ("pdf", extract_text_from_pdf),
        ("docx", extract_text_from_docx),
        ("raw", try_raw_text),
    ]
    for file_type, extract_func in methods:
        try:
            text = " ".join(filter_valid_words(extract_func(file_path)))
            # If extraction returns text without raising an exception, return it.
            if text and text.strip():
                return text
        except Exception:
            pass
    raise ValueError(
        "Unable to extract text from binary file using any method."
    )


async def fetch_pdf_if_applicable(url, timeout_value=HTTP_TIMEOUT):
    head_timeout = aiohttp.ClientTimeout(total=timeout_value)
    async with aiohttp.ClientSession(timeout=head_timeout) as session:
        is_pdf_flag = await is_pdf(session, url, timeout_value)
        if is_pdf_flag:
            # If the file is a PDF, download and parse using
            # _download_pdf function.
            try:
                text = await _download_pdf(session, url)
                if sanitized_text := sanitize_text(text):
                    BROWSER_CACHE[url] = sanitized_text
                    return sanitized_text
            except Exception:
                raise RuntimeError(
                    f"Requested to download download/parse PDF at {url} "
                    f"but failed"
                )
        return None


def attempt_download(page, url):
    with page.expect_download(timeout=5000) as download_info:
        try:
            # Attempt navigation with a wait_until that doesn't wait for full load.
            page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            # Check if the error is due to an aborted navigation.
            print(f"processing exception: {e}")
            if "ERR_ABORTED" in str(e):
                # This error is expected when the download starts.
                pass
                # print(
                #     "Navigation was aborted; likely because a download was triggered."
                # )
            else:
                raise  # If it's another error, re-raise it.
    LOGGER.debug("Download triggered!")
    download = download_info.value  # Get the Download object.
    download_path = download.path()
    # process this path, if it's a docx open, or pdf open, or...
    with open(download_path, "r") as file:
        print(file.read())
    print(f"Downloaded file at: {download_path}")


async def fetch_page_content(browser_semaphore, browser_context, url):
    try:
        content = None
        page = None
        if url in BROWSER_CACHE:
            return sanitize_text(BROWSER_CACHE[url])

        domain = urlparse(url).netloc
        domain_sem = DOMAIN_SEMAPHORES[domain]

        async with domain_sem:
            async with browser_semaphore:
                try:
                    page = await browser_context.new_page()
                    await stealth_async(page)
                    async with page.expect_download(
                        timeout=5000  # timeout in ms
                    ) as download_info:
                        await page.goto(url, wait_until="domcontentloaded")
                        download = await download_info.value
                        download_path = await download.path()
                        content = extract_text_from_binary_file(download_path)
                        BROWSER_CACHE[url] = sanitize_text(content)
                except playwright.async_api.TimeoutError:
                    await page.goto(url)
                    content = await asyncio.wait_for(
                        page.content(), timeout=5000  # timeout in ms
                    )
                    BROWSER_CACHE[url] = sanitize_text(content)
                except playwright.async_api.Error as e:
                    if "ERR_ABORTED" in str(e):
                        download = await download_info.value
                        download_path = await download.path()
                        content = extract_text_from_binary_file(download_path)
                        BROWSER_CACHE[url] = sanitize_text(content)
                    else:
                        async with httpx.AsyncClient(timeout=5) as client:

                            resp = await client.get(url)
                            resp.raise_for_status()
                            content = extract_text_from_binary_file(
                                resp.content
                            )
                            BROWSER_CACHE[url] = sanitize_text(content)
                finally:
                    if page is not None:
                        await page.close()
                return BROWSER_CACHE[url]
    except Exception:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_filename = f"error_log_{timestamp}.txt"
        with open(log_filename, "w") as f:
            f.write("Arguments:\n")
            f.write(f"browser_semaphore: {browser_semaphore}\n")
            f.write(f"browser_context: {browser_context}\n")
            f.write(f"url: {url}\n")
            f.write("Stack Trace:\n")
            f.write(traceback.format_exc())
        # LOGGER.exception(f"fetch page content failed with {url}")
        raise


def extract_text_elements(content):
    try:
        soup = BeautifulSoup(content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ", strip=True)
        clean_text = " ".join(text.split())
        if clean_text:
            return clean_text
    except Exception:
        pass
    # fallback if HTML parsing doesn't yield results
    return "".join(ch if ch.isprintable() else " " for ch in content).strip()


def cache_key(data):
    return hashlib.md5(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def count_message_tokens(messages):
    total_tokens = 0
    for m in messages:
        # role and content both count; adjust as needed for your exact usage
        total_tokens += len(GPT4O_MINI_ENCODER.encode(m["content"]))
    return total_tokens


def chunk_text_by_tokens(text, max_chunk_tokens, tokenizer):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_chunk_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks


def _make_request_with_backoff(chat_args, max_retries=5):
    openai_context = create_openai_context()
    context_window = 120000
    max_output_tokens = 16384
    overhead_tokens = 1000
    max_chunk_tokens = context_window - max_output_tokens - overhead_tokens
    if max_chunk_tokens < 1:
        raise ValueError(
            "context_window minus overhead is too small for any chunk."
        )

    messages = chat_args["messages"]

    dev_message = next((m for m in messages if m["role"] == "developer"), None)
    user_message = next((m for m in messages if m["role"] == "user"), None)
    assistant_message = next(
        (m for m in messages if m["role"] == "assistant"), None
    )
    if not any((dev_message, user_message, assistant_message)):
        raise ValueError("Developer, user, or assistant message not found.")

    token_counts = {}
    if dev_message:
        token_counts["developer"] = len(
            GPT4O_MINI_ENCODER.encode(dev_message["content"])
        )
    if user_message:
        token_counts["user"] = len(
            GPT4O_MINI_ENCODER.encode(user_message["content"])
        )
    if assistant_message:
        token_counts["assistant"] = len(
            GPT4O_MINI_ENCODER.encode(assistant_message["content"])
        )

    longest_role = max(token_counts, key=token_counts.get)
    if longest_role == "developer":
        longest_message = dev_message
    elif longest_role == "user":
        longest_message = user_message
    else:
        longest_message = assistant_message

    chunks = chunk_text_by_tokens(
        longest_message["content"], max_chunk_tokens, GPT4O_MINI_ENCODER
    )
    chunked_messages = []
    for chunk in chunks:
        message = []
        # Maintain original ordering: developer, user, assistant.
        if longest_role == "developer":
            message.append({"role": "developer", "content": chunk})
        else:
            message.append(dev_message)
        if longest_role == "user":
            message.append({"role": "user", "content": chunk})
        else:
            message.append(user_message)
        if longest_role == "assistant":
            message.append({"role": "assistant", "content": chunk})
        else:
            message.append(assistant_message)
        chunked_messages.append(message)

    full_response = []
    for messages in chunked_messages:
        backoff = 1
        args = {
            "messages": messages,
            "model": chat_args["model"],
        }

        for attempt in range(1, max_retries + 1):
            try:
                # LOGGER.debug(args)
                # LOGGER.info(
                #     f"trying with {len(args['messages'][1]['content'])} "
                #     f"characters"
                # )
                # LOGGER.info("waiting that thread")
                logging.basicConfig(level=logging.DEBUG)
                response = openai_context["client"].chat.completions.create(
                    **args
                )
                # LOGGER.info("got a response")
                full_response.append(response.choices[0].message.content)
                break
            except Exception:
                LOGGER.exception(
                    "OPENAI NOT RAISED: but openai exception, backing it off"
                )
                traceback.print_exc()
                if attempt == max_retries:
                    raise
                time.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, MAX_BACKOFF_WAIT)
    # LOGGER.info("cleaning text!")
    clean_text = sanitize_text(", ".join(full_response))
    # LOGGER.info("cleaned text!")
    openai_context["client"].close()
    return clean_text


async def get_webpage_answers(
    openai_semaphore,
    answer_context,
    full_query,
    source_url,
):
    try:
        messages = [
            {
                "role": "developer",
                "content": "You are given a snippet of text from a webpage that showed up in a Google search after querying the user question below. If you can find a specific succinct answer to that question return that answer with no other text. If there are multiple answers, return them all, comma separated. If no answer is found return UNKNOWN. Respond with no additional text besides the brief answer (s) or UNKNOWN.",
            },
            {"role": "user", "content": full_query},
            {
                "role": "assistant",
                "content": f"RELEVANT WEBPAGE SNIPPET: {answer_context}",
            },
        ]

        response_text = await generate_text(
            openai_semaphore,
            "gpt-4o-mini",
            messages,
            source_url,
        )

        # LOGGER.info(f"got that response text: {response_text}")
        return response_text
    except Exception:
        LOGGER.exception(f"error on {full_query}")


def sanitize_text(text):
    # Encode to ASCII and ignore errors, then decode back to ASCII
    # This drops any characters that cannot be mapped
    if text is None:
        return None
    return text.encode("ascii", errors="ignore").decode(
        "ascii", errors="ignore"
    )


async def generate_text(openai_semaphore, model, messages, source_url=None):
    chat_args = {"model": model, "messages": messages}
    key = cache_key(chat_args)
    if key in PROMPT_CACHE and (
        response_text := sanitize_text(PROMPT_CACHE[key])
    ):
        pass
    else:

        async with openai_semaphore:
            # LOGGER.info("to thread")
            response_text = await asyncio.to_thread(
                _make_request_with_backoff, chat_args
            )
            # LOGGER.info("got response")
        # LOGGER.info("out of _semaphore")

        PROMPT_CACHE[key] = response_text
        # LOGGER.info(f"{key}: {response_text}")
    return response_text


async def aggregate_answers(openai_semaphore, answer_list):
    try:
        combined_answers = "; ".join(answer_list)
        messages = [
            {
                "role": "developer",
                "content": (
                    "You are given a list of answers from a question answering system. "
                    "They may contain duplicates, partial overlaps, or the placeholder UNKNOWN. "
                    "Combine them into a concise, readable response. "
                    "Here is the required behavior: \n"
                    "1. Remove duplicates.\n"
                    '2. If there are any "UNKNOWN" entries, omit them unless every entry is "UNKNOWN".\n'
                    '3. If after removing UNKNOWN entries there is nothing left, return "UNKNOWN".\n'
                    '4. If there are any "FAILED" entries, omit them.'
                    "5. The final response can contain multiple words, but keep it succinct.\n"
                    "6. Do not add extra commentary or explanation, just the concise result."
                ),
            },
            {
                "role": "user",
                "content": "Combine multiple answers into a simpler one.",
            },
            {
                "role": "assistant",
                "content": f"Combined answers: {combined_answers}",
            },
        ]
        consolidated_answer = await generate_text(
            openai_semaphore, "gpt-4o-mini", messages
        )
        return consolidated_answer
    except Exception:
        LOGGER.exception(
            f"failure on aggregate answers this was the asnwer list: {answer_list}"
        )
        raise


async def main():
    parser = argparse.ArgumentParser(
        description="Process a list of queries from a file."
    )
    parser.add_argument(
        "species_list_file",
        type=argparse.FileType("r"),
        help="Path to txt file listing all species",
    )
    parser.add_argument(
        "question_list",
        type=argparse.FileType("r"),
        help="Path to the file containing query subjects of the form [header]: [question] on each line",
    )
    parser.add_argument(
        "--max_species",
        type=int,
        help="limit to this many species for debugging reasons",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        help="limit to this many subjects for debugging reasons",
    )
    parser.add_argument(
        "--google_search",
        action="store_true",
        help="If set, does the google search, else exclusively uses the cache",
    )
    parser.add_argument(
        "--web_fetch",
        action="store_true",
        help=(
            "If set, does the browser fetch with links, else exclusively "
            "uses the cache",
        ),
    )
    parser.add_argument(
        "--do_llm_scrub",
        action="store_true",
        help="Scrub the browser results so they can be read.",
    )
    parser.add_argument("--headless_off", action="store_true")
    args = parser.parse_args()

    LOGGER.info("build up species list")
    species_list = list(
        islice(
            (
                stripped_line
                for line in args.species_list_file.read().splitlines()
                if (stripped_line := line.strip())
            ),
            args.max_species,
        )
    )

    # the questions coming in are separated by a ':' representing
    # SCOPE: QUESTION, so formatting the list like that
    LOGGER.info("build up scope list")
    scope_question_list = list(
        islice(
            (
                stripped_line.split(":")
                for line in args.question_list.read().splitlines()
                if (stripped_line := line.strip())
            ),
            args.max_questions,
        )
    )

    LOGGER.info("build up species question list")
    species_question_list = [
        {
            "species": species,
            "question_body": question_body.strip(),
            "question_scope": question_scope.strip(),
            "question_formatted": question_body.format(species=species).strip(),
        }
        for question_scope, question_body in scope_question_list
        for species in species_list
    ]

    if args.google_search:
        LOGGER.info("do google searches")
        google_search_semaphore = asyncio.Semaphore(GOOGLE_QUERIES_PER_SECOND)

        async def google_search_with_progress(question_payload, pbar):
            result = await google_custom_search_async(
                google_search_semaphore,
                API_KEY,
                CX,
                question_payload["question_formatted"],
                num_results=25,
            )
            pbar.update(1)
            return {**question_payload, "search_result": result}

        pbar = tqdm(total=len(species_question_list), desc="Google Searches")
        tasks = [
            asyncio.create_task(google_search_with_progress(payload, pbar))
            for payload in species_question_list
        ]

        species_question_search_list = await asyncio.gather(*tasks)
        pbar.close()

    if args.web_fetch:
        LOGGER.info("do web fetch")
        total_fetches = sum(
            len(payload["search_result"])
            for payload in species_question_search_list
        )
        fetch_pbar = tqdm(total=total_fetches, desc="Fetching Web Pages")

        async def fetch_page_content_with_progress(
            browser_semaphore, browser_context, url, pbar
        ):
            async with browser_semaphore:
                if url in BROWSER_CACHE:
                    pbar.update(1)
                    return BROWSER_CACHE[url]
                else:
                    # LOGGER.warning(f"Missing {url}")
                    pbar.update(1)
                    return ""
            # result = await fetch_page_content(
            #     browser_semaphore, browser_context, url
            # )
            # pbar.update(1)
            # return result

        browser_semaphore = asyncio.Semaphore(MAX_TABS)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=not args.headless_off
            )
            browser_context = await browser.new_context(
                accept_downloads=False,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )

            async def search_links_from_question(question_payload):
                fetch_task_list = []
                for search_result in question_payload["search_result"]:
                    fetch_task = fetch_page_content_with_progress(
                        browser_semaphore,
                        browser_context,
                        search_result["link"],
                        fetch_pbar,
                    )
                    fetch_task_list.append(fetch_task)
                question_payload["webpage_content"] = await asyncio.gather(
                    *fetch_task_list
                )

            question_tasks = [
                asyncio.create_task(search_links_from_question(payload))
                for payload in species_question_search_list
            ]

            await asyncio.gather(*question_tasks)
            await browser_context.close()

    if args.do_llm_scrub:
        LOGGER.info("LLM scrub of webpages")

        async def scrub_url_result(url, openai_semaphore, progress_bar):
            try:
                if url in SCRUBBED_OPENAI_CACHE:
                    progress_bar.update(1)
                    return SCRUBBED_OPENAI_CACHE[url]
                if url in PROCESS_URL_CACHE:
                    messages = PROCESS_URL_CACHE[url]
                else:
                    raw_text = BROWSER_CACHE[url].strip()
                    scrubbed_text = extract_text_elements(raw_text)
                    if scrubbed_text:
                        messages = [
                            {
                                "role": "developer",
                                "content": (
                                    "Carefully remove any HTML tags, stray symbols, random"
                                    "sequences of characters, or invalid text while keeping all"
                                    "properly spelled words and punctuation intact. Do not remove or"
                                    "alter legitimate words or punctuation. Return only the cleaned"
                                    "text. Do not make up new text."
                                ),
                            },
                            {
                                "role": "user",
                                # limit us to 8,000 characters, that's about 4 pages of single spaced text
                                "content": f"{scrubbed_text[:8000]}",
                            },
                            {
                                "role": "assistant",
                                "content": "..",
                            },
                        ]
                        PROCESS_URL_CACHE[url] = messages
                    else:
                        # there's no text, just skip
                        SCRUBBED_OPENAI_CACHE[url] = ""
                        progress_bar.update(1)
                        return ""

                cleaned_text = await generate_text(
                    openai_semaphore, "gpt-4o-mini", messages
                )
                SCRUBBED_OPENAI_CACHE[url] = cleaned_text
                progress_bar.update(1)
                return cleaned_text
            except Exception:
                LOGGER.exception(f"error on {url}")
                raise

        openai_semaphore = asyncio.Semaphore(25)
        urls = list(BROWSER_CACHE.keys())
        LOGGER.debug(f"process {len(urls)}")
        progress_bar = tqdm(total=len(urls), desc="OpenAI queries")
        tasks = [
            asyncio.create_task(
                scrub_url_result(url, openai_semaphore, progress_bar)
            )
            for url in urls
        ]
        LOGGER.info("about to gather")
        _ = await asyncio.gather(*tasks)

    openai_semaphore = asyncio.Semaphore(25)
    answer_task_list = []
    answer_scope_list = []
    for index, question_payload in enumerate(species_question_search_list):
        for webpage_content, search_result in zip(
            question_payload["webpage_content"],
            question_payload["search_result"],
        ):
            url = search_result["link"]
            if url in SCRUBBED_OPENAI_CACHE:
                webpage_content = (
                    f"(MIGHT BE IRRELEVANT): {SCRUBBED_OPENAI_CACHE[url]}"
                )
            else:
                webpage_content = "IRRELEVANT FROM HERE -- NO WEBPAGE IN CACHE"
            answer_task = get_webpage_answers(
                openai_semaphore,
                (
                    f"SNIPPET: {search_result['snippet']}. FETCHED WEBPAGE "
                    f"{webpage_content}"
                ),
                f"{question_payload['question_formatted']}",
                url,
            )
            answer_task_list.append(answer_task)

            answer_scope = {
                "question_scope": question_payload["question_scope"],
                "species": question_payload["species"],
                "question_body": question_payload["question_body"],
                "question_formatted": question_payload["question_formatted"],
                "link": search_result["link"],
                "title": search_result["title"],
                "snippet": search_result["snippet"],
                "scrubbed_webpage_content": (
                    "COULD NOT READ"
                    if url not in SCRUBBED_OPENAI_CACHE
                    else SCRUBBED_OPENAI_CACHE[url][:500]
                ),
            }
            answer_scope_list.append(answer_scope)

    answers = await asyncio.gather(*answer_task_list)
    answers_by_scope_species_question = defaultdict(list)
    for answer, scope in zip(answers, answer_scope_list):
        answers_by_scope_species_question[
            (
                scope["question_scope"],
                scope["species"],
                scope["question_formatted"],
            )
        ].append(answer)
        scope["answer"] = answer

    answer_task_lookup = {}
    for key, answer_list in answers_by_scope_species_question.items():
        aggregate_answers_task = aggregate_answers(
            openai_semaphore, answer_list
        )
        answer_task_lookup[key + (tuple(answer_list),)] = aggregate_answers_task

    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def rank_answers(answer, answer_list):
        answers = [a.strip() for a in answer.split(",")]
        ranked = []
        for ans in answers:
            norm_ans = remove_punctuation(ans.lower())
            count = 0
            for potential in answer_list:
                potential_answers = [
                    remove_punctuation(x.strip().lower())
                    for x in potential.split(",")
                ]
                count += potential_answers.count(norm_ans)
            ranked.append((ans, count))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ", ".join(
            f"{remove_punctuation(ans.lower())} ({count})"
            for ans, count in ranked
        )

    answer_table = []
    for key, answer in zip(
        answer_task_lookup,
        await asyncio.gather(*answer_task_lookup.values()),
    ):
        scope, species, question, answer_list = key
        answer_table.append(
            {
                "scope": scope,
                "species": species,
                "question": question,
                "answer": rank_answers(answer, answer_list),
            }
        )

    answer_df = pd.DataFrame(answer_table)
    answer_df.to_csv("answers.csv", index=False)

    scope_df = pd.DataFrame(answer_scope_list)
    scope_df.to_csv("answer_scope.csv", index=False)


if __name__ == "__main__":
    LOGGER.info("Starting!")
    asyncio.run(main())
    LOGGER.info("All done!")
