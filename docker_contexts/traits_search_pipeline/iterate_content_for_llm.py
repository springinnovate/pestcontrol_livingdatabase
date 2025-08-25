"""Iterate through unanswered questions to answer them."""

from __future__ import annotations

from typing import Iterator, Mapping

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import (
    DB_URI,
    Question,
    QuestionKeyword,
    KeywordQuery,
    SearchHead,
    SearchResultLink,
    Link,
    Content,
    Answer,
)


def apply_llm(question_text: str, content_text: str) -> str:
    raise NotImplementedError("implement LLM call here")


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
    answer_text: str,
    dedupe: bool = True,
) -> int | None:
    if not answer_text:
        return None
    if dedupe:
        exists_stmt = select(Answer.id).where(
            Answer.keyword_query_id == keyword_query_id,
            Answer.link_id == link_id,
            Answer.answer_text == answer_text,
        )
        existing_id = session.execute(exists_stmt).scalar_one_or_none()
        if existing_id is not None:
            return existing_id
    ans = Answer(
        keyword_query_id=keyword_query_id,
        link_id=link_id,
        answer_text=answer_text,
    )
    session.add(ans)
    session.flush()
    return ans.id


def generate_and_store_answers(
    batch_commit: int = 50, dry_run: bool = False
) -> None:
    engine = create_engine(DB_URI)
    with Session(engine) as session:
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
            if not answer or dry_run:
                continue

            inserted_id = insert_answer(session, kid, lid, answer, dedupe=True)
            if inserted_id is not None:
                pending += 1
                if pending >= batch_commit:
                    session.commit()
                    pending = 0

        if pending:
            session.commit()


if __name__ == "__main__":
    generate_and_store_answers()
