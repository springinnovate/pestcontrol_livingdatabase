"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

import asyncio
import re
import json
import itertools
from typing import Iterator, Mapping, List, Dict, Any, Tuple

from sqlalchemy import create_engine, select
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


DB_ENGINE = create_engine(DB_URI)
BaseNorm.metadata.create_all(DB_ENGINE)

load_dotenv()  # loads the OPENAI_API_KEY


SYSTEM_PROMPT: str = (
    """
You are a cautious information extraction model. Given a QUESTION and the PAGE_TEXT from a single web page, decide
whether the page text supports answering the question with a strict yes/no. Only use evidence found in PAGE_TEXT.
If PAGE_TEXT does not clearly support yes or no, return unknown with a useful reason.

Output must be a single, compact JSON object with exactly these keys:
- "question": the input question (string)
- "answer": one of ["yes","no","unknown"]
- "reason": one of:
   - "supported"                # use when answer is "yes" or "no"
   - "not_enough_information"   # page has text but not enough to decide
   - "blank_page"               # PAGE_TEXT is empty/whitespace
   - "content_unrelated"        # content is unrelated to the topic/species in question
   - "conflicting_evidence"     # page states contradictory things
   - "non_english"              # content is not in English if that prevents answering
   - "page_error"               # garbled, OCR noise, or obvious fetch/render error
- "evidence": an array with 0–2 short verbatim quotes from PAGE_TEXT that justify the answer. Use [] if "unknown".
- "page_status": one of ["has_text","blank","error","non_english"] describing PAGE_TEXT itself.
- "source_url": the URL if provided, else "".

Rules:
- Answer "yes" only if PAGE_TEXT clearly asserts the proposition; answer "no" only if it clearly denies it.
- If the page mentions multiple entities, only use evidence about the entity in the question; otherwise use "unknown" with reason "content_unrelated" or "not_enough_information".
- Do not speculate or use outside knowledge.
- Keep "evidence" quotes very short (≤200 characters each), and quote verbatim from PAGE_TEXT.
- Return JSON only. No markdown, no commentary, no code fences.
""".strip()
)


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
        "model": "gpt-5",
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
    """Iterate over question–content rows where no answer exists yet.

    Returns:
        Iterator[Mapping[str, object]] with fields:
            - question_id (int)
            - question_text (str)
            - keyword_query_id (int)
            - link_id (int)
            - link_url (str)
            - content_id (int)
            - content_text (str)
    """
    stmt = (
        select(
            Question.id.label("question_id"),
            Question.text.label("question_text"),
            QuestionKeyword.keyword_query_id.label("keyword_query_id"),
            Link.id.label("link_id"),
            Link.url.label("link_url"),
            Content.id.label("content_id"),
            Content.text.label("content_text"),
        )
        .join(QuestionKeyword, QuestionKeyword.question_id == Question.id)
        .join(SearchHead, SearchHead.question_id == Question.id)
        .join(
            SearchResultLink,
            SearchResultLink.search_head_id == SearchHead.question_id,
        )
        .join(Link, Link.id == SearchResultLink.link_id)
        .join(Content, Content.id == Link.content_id, isouter=True)
        .join(
            Answer,
            (Answer.keyword_query_id == QuestionKeyword.keyword_query_id)
            & (Answer.link_id == Link.id),
            isouter=True,
        )
        .where(Content.id.is_not(None))
        .where(Answer.id.is_(None))
    ).execution_options(stream_results=True)

    for row in session.execute(stmt).mappings():
        yield row


def insert_answer(
    session: Session,
    keyword_query_id: int,
    link_id: int,
    answer: Dict[str, str],
) -> int | None:
    """Insert a new answer if there is not already one in the db."""
    exists_stmt = select(Answer.id).where(
        Answer.keyword_query_id == keyword_query_id,
        Answer.link_id == link_id,
        Answer.answer_text == answer["answer"],
    )
    existing_id = session.execute(exists_stmt).scalar_one_or_none()
    if existing_id is not None:
        return existing_id
    ans = Answer(
        keyword_query_id=keyword_query_id,
        link_id=link_id,
        answer_text=answer["answer"],
        reason=answer["reason"],
        evidence=answer["evidence"],
    )
    session.add(ans)
    session.flush()
    return ans.id


async def call_llm_async(
    question_text: str, content_text: str
) -> Dict[str, Any]:
    """Wrapper to sync LLM call in a thread."""
    return await asyncio.to_thread(apply_llm, question_text, content_text)


async def _worker(
    row: Dict[str, Any], sem: asyncio.Semaphore
) -> Tuple[Dict[str, Any], str | Dict[str, Any] | None]:
    async with sem:
        qtext: str = row["question_text"]
        content_text: str = row["content_text"] or ""
        print(
            f"\n[QUESTION #{row['question_id']}] "
            f"(KeywordQuery #{row['keyword_query_id']}, Link #{row['link_id']})\n"
            f"Q: {qtext.strip()[:120]}{'...' if len(qtext) > 120 else ''}\n"
            f"Content: {content_text.strip()[:200]}{'...' if len(content_text) > 200 else ''}\n"
            f"{'-'*80}"
        )

        # If content_text is an error marker like [[some text]], short-circuit with 'unknown - [[some text]]'
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
        except Exception as e:
            print(
                f"LLM error on question_id={row['question_id']}, link_id={row['link_id']}: {e}"
            )
            return row, None


async def run_parallel_generation(
    max_concurrency: int = 8,
    batch_commit: int = 50,
) -> None:
    engine = create_engine(DB_URI)

    # snapshot rows first (so DB cursor isn't shared across async tasks)
    print("snapshot rows")
    with Session(engine) as session:
        # the 'None' is a placeholder so we can put a limit if we need
        unanswered_questions = list(
            itertools.islice(iter_unanswered_questions(session), None)
        )

    sem = asyncio.Semaphore(max_concurrency)
    print("creating tasks")
    tasks = [
        asyncio.create_task(_worker(question, sem))
        for question in unanswered_questions
    ]

    pending_inserts: List[Tuple[int, int, Dict[str, Any]]] = []
    inserted = 0

    for fut in asyncio.as_completed(tasks):
        row, answer = await fut
        if not answer:
            continue
        kid: int = row["keyword_query_id"]
        lid: int = row["link_id"]
        pending_inserts.append((kid, lid, answer))

        if len(pending_inserts) >= batch_commit:
            with Session(engine) as session:
                for kid_i, lid_i, ans in pending_inserts:
                    insert_answer(session, kid_i, lid_i, ans)
                session.commit()
            inserted += len(pending_inserts)
            pending_inserts.clear()

    if pending_inserts:
        with Session(engine) as session:
            for kid_i, lid_i, ans in pending_inserts:
                insert_answer(session, kid_i, lid_i, ans)
            session.commit()
        inserted += len(pending_inserts)

    print(f"\nDone. Inserted {inserted} answers.")


if __name__ == "__main__":
    asyncio.run(run_parallel_generation())
