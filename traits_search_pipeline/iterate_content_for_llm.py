"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

import json
from typing import Iterator, Mapping, List, Dict

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from openai import OpenAI
from models import (
    DB_URI,
    BaseNorm,
    Question,
    QuestionKeyword,
    KeywordQuery,
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
    openai_client = OpenAI()
    openai_context = {
        "client": openai_client,
    }
    return openai_context


def apply_llm(question: str, page_text: str) -> List[Dict[str, str]]:
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
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },  # if supported in your SDK/runtime
        "messages": messages,
    }
    openai_context = create_openai_context()
    resp = openai_context["client"].chat.completions.create(**args)
    content = json.loads(resp.choices[0].message.content)
    return content


def iter_question_content_rows(
    session: Session,
) -> Iterator[Mapping[str, object]]:
    # question -> question_keyword -> keyword_query
    # question -> search_head -> search_result_link -> link -> content
    stmt = (
        select(
            Question.id.label("question_id"),
            Question.text.label("question_text"),
            KeywordQuery.id.label("keyword_query_id"),
            Link.id.label("link_id"),
            Link.url.label("link_url"),
            Content.id.label("content_id"),
            Content.text.label("content_text"),
        )
        .join(QuestionKeyword, QuestionKeyword.question_id == Question.id)
        .join(KeywordQuery, KeywordQuery.id == QuestionKeyword.keyword_query_id)
        .join(SearchHead, SearchHead.question_id == Question.id)
        .join(
            SearchResultLink,
            SearchResultLink.search_head_id == SearchHead.question_id,
        )
        .join(Link, Link.id == SearchResultLink.link_id)
        .join(Content, Content.id == Link.content_id, isouter=True)
        .where(Content.id.is_not(None))
    ).execution_options(stream_results=True)

    for row in session.execute(stmt).mappings():
        yield row


def iter_question_content_pairs(session: Session) -> Iterator[tuple[str, str]]:
    for row in iter_question_content_rows(session):
        yield row["question_text"], row["content_text"]  # convenient 2-tuple


def insert_answer(
    session: Session,
    keyword_query_id: int,
    link_id: int,
    answer: Dict[str, str],
) -> int | None:
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


def generate_and_store_answers(batch_commit: int = 10) -> None:
    with Session(DB_ENGINE) as session:
        pending = 0
        for row in iter_question_content_rows(session):
            qtext: str = row["question_text"]
            content_text: str = row["content_text"]
            kid: int = row["keyword_query_id"]
            lid: int = row["link_id"]

            print(
                f"\n[QUESTION #{row['question_id']}] "
                f"(KeywordQuery #{kid}, Link #{lid})\n"
                f"Q: {qtext.strip()[:120]}{'...' if len(qtext) > 120 else ''}\n"
                f"Content: {content_text.strip()[:200]}{'...' if len(content_text) > 200 else ''}\n"
                f"{'-'*80}"
            )

            answer = apply_llm(qtext, content_text)
            print(answer)
            if not answer:
                print("warning no answer!")
                return

            inserted_id = insert_answer(session, kid, lid, answer)
            if inserted_id is not None:
                pending += 1
                if pending >= batch_commit:
                    session.commit()
                    pending = 0
                    break

        if pending:
            session.commit()


if __name__ == "__main__":
    generate_and_store_answers()
