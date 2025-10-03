"""Iterate through unanswered questions to answer them.

docker build -t pcld_trait_search_env:latest . && docker run --rm -it --gpus all -v "%CD%":/app -w /app pcld_trait_search_env:latest
python answer_questions_with_content.py


"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import deque
from multiprocessing import Manager
from typing import Any
from queue import Empty
import logging
import re
import time
from typing import List, Tuple, Optional

from sqlalchemy import create_engine, select, func, and_, or_
from sqlalchemy.orm import Session
import psutil

from models import (
    DB_URI,
    BaseNorm,
    Content,
    Answer,
    QuestionLink,
    Question,
    Link,
)

from dotenv import load_dotenv
from logging_tools import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)
from llm_tools import apply_llm, truncate_to_token_limit
from huggingface_tools import generate, GenConfig

DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)

GLOBAL_LOG_LEVEL = logging.DEBUG


load_dotenv()  # loads the OPENAI_API_KEY

OPENAI_MODEL = "gpt-4.1-mini"


SYSTEM_PROMPT = """
You are a cautious information extraction model.

Given a QUESTION and a CONTEXT_WINDOW (excerpted text from a single web page; optionally SOURCE_URL), decide the answer using ONLY the CONTEXT_WINDOW (you may use the QUESTION for disambiguation rules below). Do NOT use outside knowledge. Do NOT speculate.

STRICT FORMAT
Return ONE compact JSON object with EXACTLY these top-level keys and NOTHING ELSE:
- "answer_text": string
- "reason": string (enum below)
- "evidence": array of 0–2 strings

Compliance rules:
- Keys MUST be exactly as above. NEVER use "answer", "answerKey", or any other key.
- If you cannot comply, output: {"answer_text":"unknown","reason":"page_error","evidence":[]}
- Output JSON only. No markdown, no comments, no code fences, no trailing commas.

"reason" enum:
["supported","not_enough_information","blank_page","content_unrelated","conflicting_evidence","non_english","page_error","model_uncertain"]

ALLOWED VALUES FOR "answer_text" (infer task type from QUESTION)
- Predator life stage: "larvae"|"nymphs"|"adults"|"larvae,nymphs"|"larvae,adults"|"nymphs,adults"|"larvae,nymphs,adults"|"unknown"
- Agricultural pest life stage: same set as above
- Invasiveness: "yes"|"no"|"unknown"
- Non-crop habitat use: "yes"|"no"|"unknown"
- Dietary breadth: "specialist"|"generalist"|"unknown"
- Wind dispersal: "yes"|"no"|"unknown"
- Flight ability: "yes"|"no"|"unknown"
- Seasonal migration: "yes"|"no"|"unknown"
- Pest status: "yes"|"no"|"unknown"

DECISION RULES
- Treat CONTEXT_WINDOW as a partial snippet: absence of evidence outside this snippet MUST NOT be assumed. Default to {"answer_text":"unknown","reason":"not_enough_information"} unless the CONTEXT_WINDOW clearly supports an allowed value; then "reason":"supported".
- Species abbreviation handling: If either the QUESTION or the CONTEXT_WINDOW introduces a species by its full genus + species name (e.g., "Acalymma vittatum"), you may treat later occurrences of the abbreviated form with the same initial (e.g., "A. vittatum") or with no initial at all in the CONTEXT_WINDOW as the same entity.
- Coreference within snippet: pronouns like "it", "the species", and repeated common names refer to the last explicitly named target species unless the text clearly switches subjects.
- Exclusivity like "nymphs only" -> that stage.
- Multiple supported stages -> comma-join in canonical order: larvae,nymphs,adults.
- For "diet": narrow range -> "specialist"; broad/diverse -> "generalist"; explicit contradictions -> {"answer_text":"unknown","reason":"conflicting_evidence"}.
- "can fly" refers to adult flight unless clearly specified otherwise.
- If CONTEXT_WINDOW is about a different entity -> "content_unrelated".
- Mentions only higher/lower taxa without explicit linkage to the target species -> "not_enough_information".
- Non-English blocking judgment -> "non_english".
- Blank/garbled/unusable snippet -> "blank_page" or "page_error" as applicable.

EVIDENCE RULES
- Provide up to 2 short verbatim quotes (≤200 chars) copied exactly from CONTEXT_WINDOW that directly support "answer_text".
- NEVER include processing markers or placeholders (e.g., [[...]], <...>, "PDF_SKIPPED", "TRUNCATED"). If the snippet shows truncation or is otherwise unusable -> "evidence":[] and "reason":"page_error".
- Use [] when "answer_text" is "unknown".

NORMALIZATION
- Lowercase "answer_text".
- Use only allowed strings.
- Output exactly three keys as specified.

VALID EXAMPLE
{"answer_text":"yes","reason":"supported","evidence":["...exact quote 1...","...exact quote 2..."]}

UNKNOWN EXAMPLE
{"answer_text":"unknown","reason":"not_enough_information","evidence":[]}
""".strip()


def process_unanswered_questions() -> None:
    """Orchestrate parallel LLM answering for unanswered (question, link) pairs.

    Spawns a process pool of workers to generate answers with an LLM for each
    (question, link) that lacks an Answer, and a dedicated writer process that
    persists results into the database. Uses inter-process queues for task and
    result passing, and a shared event to coordinate shutdown.

    Steps:
      * Initialize process-safe logging and shared IPC primitives.
      * Query unanswered QuestionLink/Link pairs with non-null Content.
      * Enqueue tasks to worker processes.
      * Collect model outputs and persist them via the writer.
      * Ensure graceful shutdown on completion or error.

    Args:
        openai_model (str): a valid open AI model to use to do llm processing.

    Returns:
        None

    Raises:
        Exception: Propagates unexpected errors after attempting cleanup.
    """
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/iterate_content_for_llm.log", GLOBAL_LOG_LEVEL
    )
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(logger)
    question_id_link_id_queue = manager.Queue()
    result_answer_question_id_link_id_queue = manager.Queue()
    stop_processing_event = manager.Event()
    try:
        logger.info("start workers")
        if OPENAI_MODEL is None:
            n_workers = 1
        else:
            n_workers = psutil.cpu_count(logical=False)
        with ProcessPoolExecutor(
            max_workers=n_workers + 1
        ) as pool:  # +1 for the db_writer
            worker_futures = [
                pool.submit(
                    _answer_question_worker,
                    question_id_link_id_queue,
                    result_answer_question_id_link_id_queue,
                    stop_processing_event,
                    log_queue,
                )
                for _ in range(n_workers)
            ]
            answer_insert_future = pool.submit(
                _insert_answer_worker,
                result_answer_question_id_link_id_queue,
                stop_processing_event,
                log_queue,
            )
            logger.info("queue up the questions")

            with Session(DB_ENGINE) as session:
                stmt = (
                    select(QuestionLink, Link)
                    .join_from(QuestionLink, Link, QuestionLink.link)
                    .join(Content, Content.id == Link.content_id)
                    .outerjoin(
                        Answer,
                        and_(
                            Answer.question_id == QuestionLink.question_id,
                            Answer.link_id == QuestionLink.link_id,
                        ),
                    )
                    .where(
                        or_(Content.is_valid.is_(None), Content.is_valid != 0)
                    )
                    .group_by(
                        QuestionLink.id,
                        Link.id,
                        Link.content_id,
                        Link.url,
                        Link.fetch_error,
                        Content.id,
                        Content.is_valid,
                    )
                    .having(func.count(Answer.id) == 0)
                    .execution_options(stream_results=True, yield_per=200)
                )

                with open("questions_to-answer.txt", "w") as file:
                    file.write(
                        "question_link.id,question_link.question_id,question_link.link_id,link.content_id\n"
                    )
                    for index, (question_link, link) in enumerate(
                        session.execute(stmt)
                    ):
                        file.write(
                            f"{question_link.id},{question_link.question_id},{question_link.link_id},{link.content_id}\n"
                        )
                        question_id_link_id_queue.put(
                            (
                                question_link.id,
                                question_link.question_id,
                                question_link.link_id,
                                link.content_id,
                            )
                        )

            for _ in range(n_workers):
                question_id_link_id_queue.put(None)

            logger.info("wait for _answer_question_worker to finish")
            for f in worker_futures:
                f.result()
            logger.info("_answer_question_worker are finished, signal to stop")
            result_answer_question_id_link_id_queue.put(None)
            stop_processing_event.set()
            logger.info("await _insert_answer_worker")
            answer_insert_future.result()
            logger.info(
                "all done with process_unanswered_questions url, clean up"
            )
    finally:
        logger.info("all done with process_unanswered_questions, exiting")
        listener.stop()


def build_context_windows(
    text: str,
    key_substring: str,
    buffer_chars: int,
    overlap_threshold: float = 0.5,
    min_separation_chars: int = 100,
    max_windows: Optional[int] = None,
) -> List[str]:
    """
    Extracts buffered context windows around occurrences of `key_substring`,
    then merges highly overlapping/nearby windows to avoid redundancy.

    Args:
        text: Full source text.
        key_substring: Substring (can be multi-word). Arbitrary whitespace between words is allowed.
        buffer_chars: Number of chars to include before/after each match before merge.
        overlap_threshold: If two windows overlap by > this fraction of the smaller window, merge them.
        min_separation_chars: Also merge if the gap between windows is <= this many chars.
        max_windows: If provided, truncate the final list to this many windows.

    Returns:
        List of merged snippets with ellipses where trimmed.
    """
    # build pattern that tolerates arbitrary whitespace between words
    words = key_substring.strip().split()
    if not words:
        return []
    pattern = r"\b" + r"\s+".join(map(re.escape, words)) + r"\b"

    # collect raw [s,e) windows expanded by buffer and snapped to whitespace
    intervals: List[Tuple[int, int]] = []
    for m in re.finditer(pattern, text, flags=re.I | re.S | re.M):
        start, end = m.span()
        s = max(0, start - buffer_chars)
        e = min(len(text), end + buffer_chars)

        # expand to nearest whitespace so we don't cut mid-word
        while s > 0 and not text[s].isspace():
            s -= 1
        while e < len(text) and (e == 0 or not text[e - 1].isspace()):
            e += 1
            if e >= len(text):
                e = len(text)
                break

        intervals.append((s, e))

    if not intervals:
        return []

    # sort by start
    intervals.sort(key=lambda se: se[0])

    # helper: compute fractional overlap relative to smaller interval
    def frac_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        s1, e1 = a
        s2, e2 = b
        inter = max(0, min(e1, e2) - max(s1, s2))
        len1 = e1 - s1
        len2 = e2 - s2
        smaller = max(1, min(len1, len2))
        return inter / smaller

    # merge intervals if highly overlapping or very close
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        cur = (cur_s, cur_e)
        nxt = (s, e)
        close_enough = (s - cur_e) <= max(0, min_separation_chars)
        if frac_overlap(cur, nxt) > overlap_threshold or close_enough:
            # merge
            cur_e = max(cur_e, e)
            cur_s = min(cur_s, s)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # materialize snippets with ellipses if trimmed
    windows: List[str] = []
    for s, e in merged:
        snippet = text[s:e].strip()
        if s > 0:
            snippet = "… " + snippet
        if e < len(text):
            snippet = snippet + " …"
        windows.append(snippet)

    if max_windows is not None:
        windows = windows[:max_windows]

    return windows


def _answer_question_worker(
    question_id_link_id_queue: Any,
    result_answer_question_id_link_id_queue: Any,
    stop_processing_event: Any,
    log_queue: Any,
) -> None:
    """Worker process that generates an Answer for each (question, link, content).

    Consumes items of the form:
        (question_link_id: int, question_id: int, link_id: int, content_id: int)
    from the task queue, fetches the corresponding Question and Content rows,
    constructs a prompt, calls the LLM, and pushes a result tuple to the result
    queue:
        (question_link_id: int, question_id: int, link_id: int, result: dict)

    The worker exits when it receives a sentinel (None) or when the shared stop
    event is set.

    Args:
        question_id_link_id_queue: IPC queue to receive work items; items are tuples
            (question_link_id, question_id, link_id, content_id).
        result_answer_question_id_link_id_queue: IPC queue to emit LLM outputs as
            tuples (question_link_id, question_id, link_id, result_dict).
        stop_processing_event: Shared event used to signal a cooperative shutdown.
        log_queue: IPC queue for process-safe logging.

    Returns:
        None

    Raises:
        Exception: On unrecoverable errors; sets the stop event and logs the
            exception before exiting.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_answer_question_worker"
        )
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        last_time = 0
        ewma = None
        alpha = 0.001
        while not stop_processing_event.is_set():
            try:
                item = question_id_link_id_queue.get(timeout=5)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return
            try:
                queue_len = question_id_link_id_queue.qsize()
                current_time = time.time()
                per_item = current_time - last_time
                if ewma is None:
                    ewma = per_item
                else:
                    ewma = alpha * per_item + (1 - alpha) * ewma
                eta_seconds = max(0.0, ewma * queue_len)
                eta_str = time.strftime(
                    "%H:%M:%S", time.gmtime(int(round(eta_seconds)))
                )
                logger.info(f"{queue_len} elements left to process {eta_str}")
                last_time = current_time
            except NotImplementedError:
                logger.info("unknown number of elements left ot process")

            question_link_id, question_id, link_id, content_id = item

            with Session(DB_ENGINE) as session:
                content = session.scalar(
                    select(Content).where(Content.id == content_id)
                )
                question = session.scalar(
                    select(Question).where(Question.id == question_id)
                )
                # logger.debug(f"processing {question_link_id}")

            required_phrase_match = re.search(
                r'"([^"]+)"', question.keyword_phrase
            )
            # this just gets the last word
            quoted_phrase = (
                required_phrase_match.group(1).lower().split(" ")[-1]
            )
            normalized_content_text = re.sub(
                r"\s+", " ", content.text.lower()
            ).strip()
            if quoted_phrase not in normalized_content_text:
                logger.warning(f"{quoted_phrase} not in the context text")
                result = {
                    "answer_text": "not relevant context",
                    "reason": f"{quoted_phrase} not found in the context",
                    "evidence": [],
                    "context": normalized_content_text,
                }
                result_answer_question_id_link_id_queue.put(
                    (question_link_id, question_id, link_id, result)
                )

            context_windows = build_context_windows(
                normalized_content_text, quoted_phrase, 1500
            )

            if OPENAI_MODEL is not None:
                for index, context in enumerate(context_windows):
                    logger.info(
                        f"trying {OPENAI_MODEL} {index+1} of {len(context_windows)} for {question.text}"
                    )
                    user_prompt = (
                        f"QUESTION: {question.text} "
                        f"CONTEXT_WINDOW: {context}"
                    )
                    clean_user_prompt = truncate_to_token_limit(
                        user_prompt, OPENAI_MODEL, 10000
                    )
                    result = apply_llm(
                        SYSTEM_PROMPT, clean_user_prompt, logger, OPENAI_MODEL
                    )
                    if result:
                        # pick +/- 40 characters around the keyword as conteext
                        quoted_index = context.lower().find(
                            quoted_phrase.lower()
                        )
                        if quoted_index != -1:
                            start = max(0, quoted_index - 40)
                            end = min(len(context), quoted_index + 40)
                            snippet = context[start:end].strip()
                            if start > 0:
                                snippet = "… " + snippet
                            if end < len(context):
                                snippet = snippet + " …"
                            result["context_preview"] = snippet
                        else:
                            result["context_preview"] = context[:80]
                        result["context"] = context
                        result_answer_question_id_link_id_queue.put(
                            (question_link_id, question_id, link_id, result)
                        )
            else:
                cfg = GenConfig(
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.05,
                )

                local_model = "mistralai/Mistral-7B-Instruct-v0.3"
                for index, context in enumerate(context_windows):
                    logger.info(
                        f"trying {index+1} of {len(context_windows)} for {question.text}"
                    )
                    result = generate(
                        model_name=local_model,
                        question=question.text,
                        page_text=context,
                        device="cuda",
                        cfg=cfg,
                    )
                    if result and result["parsed"]["answer_text"] != "unknown":
                        logger.info(
                            f"found answer {index+1} of {len(context_windows)} as "
                            + result["parsed"]["answer_text"]
                            + " reason: "
                            + result["parsed"]["reason"]
                            + " evidence"
                            + str(result["parsed"]["evidence"])
                        )
                        result_answer_question_id_link_id_queue.put(
                            (
                                question_link_id,
                                question_id,
                                link_id,
                                result["parsed"],
                            )
                        )
                        break
                logger.info(
                    f"local model didn't find answer for {question.text}"
                )
                result = {
                    "answer_text": f"local model did not answer: {local_model}",
                    "reason": "local model failed",
                    "evidence": [],
                }
                result_answer_question_id_link_id_queue.put(
                    (question_link_id, question_id, link_id, result)
                )

            # generate()

            # logger.info(
            #     f"this is the phrase: {quoted_phrase} and this is the context windows for it: {context_windows}"
            # )
            # return

    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened in _answer_question_worker")


def _insert_answer_worker(
    result_answer_question_id_link_id_queue: Any,
    stop_processing_event: Any,
    log_queue: Any,
) -> None:
    """Writer process that persists Answer rows produced by workers.

    Consumes items of the form:
        (question_link_id: int, question_id: int, link_id: int, result: dict)
    where result has keys:
        - 'answer_text': str
        - 'reason': str
        - 'evidence': list
    It inserts a new Answer row per item and commits the transaction. The writer
    exits on a sentinel (None) or when the shared stop event is set.

    Args:
        result_answer_question_id_link_id_queue: IPC queue providing result tuples
            (question_link_id, question_id, link_id, result_dict).
        stop_processing_event: Shared event used to signal a cooperative shutdown.
        log_queue: IPC queue for process-safe logging.

    Returns:
        None

    Raises:
        Exception: On database or unexpected errors; sets the stop event and logs
            the exception before exiting.
    """
    try:
        logger = configure_worker_logger(
            log_queue, GLOBAL_LOG_LEVEL, "_insert_answer_worker"
        )
        while not stop_processing_event.is_set():
            try:
                item = result_answer_question_id_link_id_queue.get(timeout=10)
            except Empty:
                logger.info("task queue empty; polling again")
                continue
            if item is None:
                # sentinel, quit
                return

            question_link_id, question_id, link_id, result = item
            # result has the keys ('answer_text', 'reason', 'evidence': [])

            with Session(DB_ENGINE) as session:
                try:
                    answer = Answer(
                        question_link_id=question_link_id,
                        question_id=question_id,
                        link_id=link_id,
                        answer_text=result["answer_text"],
                        reason=result["reason"],
                        evidence=result["evidence"],
                        context=result.get("context", None),
                    )
                    session.add(answer)
                    session.commit()
                except KeyError:
                    # this keyerror happens sometimes when the LLM incorrectly
                    # formats the structured return value, so we record it
                    # but keep going
                    logger.warning(
                        f"something weird happened on this result: '{result}'"
                    )
                    answer = Answer(
                        question_link_id=question_link_id,
                        question_id=question_id,
                        link_id=link_id,
                        answer_text="error",
                        reason=str(result),
                        evidence="unknown",
                    )
                    session.add(answer)
                    session.commit()
    except Exception:
        stop_processing_event.set()
        logger.exception("something bad happened in _insert_answer_worker")


if __name__ == "__main__":
    process_unanswered_questions()
