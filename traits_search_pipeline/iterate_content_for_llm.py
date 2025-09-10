"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

from typing import Iterator, Mapping, List, Dict, Any, Tuple
import asyncio
import itertools
import json
import logging
import re
from multiprocessing import Queue

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import Session

from openai import OpenAI
from models import (
    DB_URI,
    BaseNorm,
    Question,
    QuestionKeyword,
    SearchHead,
    SearchResultLink,
    Link,
    Content,
    Answer,
)

from dotenv import load_dotenv
from url_scraper import (
    set_catch_and_log_logger,
    configure_worker_logger,
    start_process_safe_logging,
)

DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)

load_dotenv()  # loads the OPENAI_API_KEY


# 1) Harden the prompt to forbid "answer" and placeholder evidence

SYSTEM_PROMPT = """
You are a cautious information extraction model.

Given a QUESTION and PAGE_TEXT from a single web page (optionally SOURCE_URL), decide the answer using ONLY PAGE_TEXT. Do NOT speculate.

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
["supported","not_enough_information","blank_page","content_unrelated","conflicting_evidence","non_english","page_error"]

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
- Default to "answer_text":"unknown" unless PAGE_TEXT clearly supports an allowed value; then "reason":"supported".
- Exclusivity like "nymphs only" → that stage.
- Multiple supported stages → comma-join in canonical order: larvae,nymphs,adults.
- For "diet": narrow range → "specialist"; broad/diverse → "generalist"; explicit contradictions → "answer_text":"unknown","reason":"conflicting_evidence".
- "can fly" refers to adult flight unless clearly specified otherwise.
- If PAGE_TEXT is about a different entity → "content_unrelated".
- Mentions only higher/lower taxa without linkage → "not_enough_information".
- Non-English blocking judgment → "non_english".
- Blank/garbled page → "blank_page" or "page_error" as applicable.

EVIDENCE RULES
- Provide up to 2 short verbatim quotes (≤200 chars) that directly support "answer_text".
- NEVER include processing markers or placeholders (e.g., [[...]], <...>, "PDF_SKIPPED", "TRUNCATED"). If PAGE_TEXT was truncated/skipped or otherwise unusable → "evidence":[] and "reason":"page_error".
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


def create_openai_context():
    """Helper to create openai context."""
    openai_client = OpenAI()
    openai_context = {
        "client": openai_client,
    }
    return openai_context


def apply_llm(question: str, page_text: str) -> List[Dict[str, str]]:
    """Invoke the question and context on the `SYSTEM_PROMPT`."""
    user_payload = (
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "PAGE_TEXT:\n"
        f"{page_text.strip()}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    args = {
        "model": "gpt-4o-mini",
        "response_format": {
            "type": "json_object"
        },  # if supported in your SDK/runtime
        "messages": messages,
    }
    openai_context = create_openai_context()
    resp = openai_context["client"].chat.completions.create(**args)
    content = json.loads(resp.choices[0].message.content)
    return content


def iter_unanswered_questions(
    session: Session,
) -> Iterator[Mapping[str, object]]:
    stmt = (
        select(
            Question.id.label("question_id"),
            Question.text.label("question_text"),
            QuestionKeyword.keyword_query_id.label("keyword_query_id"),
            Link.id.label("link_id"),
            Link.url.label("link_url"),
            Content.id.label("content_id"),
        )
        .join(QuestionKeyword, QuestionKeyword.question_id == Question.id)
        .join(SearchHead, SearchHead.question_id == Question.id)
        .join(
            SearchResultLink,
            SearchResultLink.search_head_id == SearchHead.question_id,
        )
        .join(Link, Link.id == SearchResultLink.link_id)
        .join(Content, Content.id == Link.content_id)
        .join(
            Answer,
            (Answer.keyword_query_id == QuestionKeyword.keyword_query_id)
            & (Answer.link_id == Link.id),
            isouter=True,
        )
        .where(or_(Answer.id.is_(None), Answer.reason == "page_error"))
    ).execution_options(stream_results=True)

    for row in session.execute(stmt).mappings():
        yield row


def insert_answer(
    session: Session,
    keyword_query_id: int,
    link_id: int,
    answer: Dict[str, str],
    log_queue: Queue,
) -> int | None:
    """Insert a new answer if there is not already one in the db."""
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )
    logger.debug(f"this is the answer: {answer}")
    if "answer_text" in answer:
        answer_text = answer["answer_text"]
    elif "answer" in answer:
        answer_text = answer["answer"]
    else:
        raise ValueError(f"expected some sort of answer key but got {answer}")
    exists_stmt = select(Answer.id).where(
        Answer.keyword_query_id == keyword_query_id,
        Answer.link_id == link_id,
        Answer.answer_text == answer_text,
    )
    existing_id = session.execute(exists_stmt).scalar_one_or_none()
    if existing_id is not None:
        logger.info(f"statement exists at {existing_id}")
        return existing_id
    ans = Answer(
        keyword_query_id=keyword_query_id,
        link_id=link_id,
        answer_text=answer_text,
        reason=answer["reason"],
        evidence=answer["evidence"],
    )
    session.add(ans)
    session.flush()
    logger.info(f"statement inserted at {ans.id}")
    return ans.id


async def call_llm_async(
    question_text: str, content_text: str
) -> Dict[str, Any]:
    """Wrapper to sync LLM call in a thread."""
    return await asyncio.to_thread(apply_llm, question_text, content_text)


async def _worker(
    row: Dict[str, Any], sem: asyncio.Semaphore, log_queue: Queue
) -> Tuple[Dict[str, Any], str | Dict[str, Any] | None]:
    logger = configure_worker_logger(
        log_queue, GLOBAL_LOG_LEVEL, "_url_scrape_worker"
    )

    async with sem:
        qtext: str = row["question_text"]
        content_text: str = row["content_text"] or ""
        logger.info(
            f"\n[QUESTION #{row['question_id']}] "
            f"(KeywordQuery #{row['keyword_query_id']}, Link #{row['link_id']})\n"
            f"Q: {qtext.strip()[:120]}{'...' if len(qtext) > 120 else ''}\n"
            f"Content: {content_text.strip()[:200]}{'...' if len(content_text) > 200 else ''}\n"
            f"{'-'*80}"
        )

        m = re.search(r"\[\[(.*?)\]\]", content_text, flags=re.DOTALL)
        if m:
            err_text = m.group(1).strip()
            answer = {
                "answer": "unknown",
                "reason": "page_error",
                "evidence": [f"[[{err_text}]]"],
            }
            return row, answer

        try:
            answer = await call_llm_async(qtext, content_text)
            return row, answer
        except asyncio.CancelledError:
            logger.info(
                f'task cancelled for question_id={row["question_id"]}, link_id={row["link_id"]}'
            )
            raise
        except Exception as e:
            logger.exception(
                f'LLM error on question_id={row["question_id"]}, link_id={row["link_id"]}: {e}'
            )
            raise


def get_content_text(
    session: Session, question_context: Dict[str, str]
) -> str | None:
    result = dict(question_context)
    result["content_text"] = session.execute(
        select(Content.text).where(Content.id == question_context["content_id"])
    ).scalar_one_or_none()
    return result


async def run_parallel_generation(
    max_concurrency: int = 1,
    batch_commit: int = 50,
) -> None:
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = logging.DEBUG
    log_queue, listener, manager = start_process_safe_logging(
        "logs/iterate_content_for_llm.log"
    )
    main_logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    set_catch_and_log_logger(main_logger)

    engine = create_engine(DB_URI)

    main_logger.info("snapshot rows")
    with Session(engine) as session:
        unanswered_questions = list(
            itertools.islice(iter_unanswered_questions(session), None)
        )
        main_logger.info(unanswered_questions)
        sem = asyncio.Semaphore(max_concurrency)
        main_logger.info("creating tasks")
        tasks = [
            asyncio.create_task(
                _worker(get_content_text(session, question), sem, log_queue)
            )
            for question in unanswered_questions
        ]

    pending_inserts: List[Tuple[int, int, Dict[str, Any]]] = []
    inserted = 0

    try:
        for fut in asyncio.as_completed(tasks):
            try:
                row, answer = await fut
            except Exception as e:
                # cancel all remaining tasks and wait for them to finish
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                main_logger.error(f"aborting due to error: {e}")
                # optional: flush any already batched inserts before aborting
                if pending_inserts:
                    with Session(engine) as session:
                        for kid_i, lid_i, ans in pending_inserts:
                            insert_answer(session, kid_i, lid_i, ans, log_queue)
                        session.commit()
                    inserted += len(pending_inserts)
                    pending_inserts.clear()
                raise

            if not answer:
                continue
            kid: int = row["keyword_query_id"]
            lid: int = row["link_id"]
            main_logger.debug(f"this is the answer: {answer}")
            pending_inserts.append((kid, lid, answer))

            if len(pending_inserts) >= batch_commit:
                with Session(engine) as session:
                    for kid_i, lid_i, ans in pending_inserts:
                        insert_answer(session, kid_i, lid_i, ans, log_queue)
                    session.commit()
                inserted += len(pending_inserts)
                pending_inserts.clear()

        if pending_inserts:
            with Session(engine) as session:
                for kid_i, lid_i, ans in pending_inserts:
                    insert_answer(session, kid_i, lid_i, ans, log_queue)
                session.commit()
            inserted += len(pending_inserts)

        main_logger.info(f"\nDone. Inserted {inserted} answers.")
    finally:
        listener.stop()
        logging.shutdown()


if __name__ == "__main__":
    asyncio.run(run_parallel_generation())
