"""Generate a species–question–answer report from the database.

This script queries all Species, their associated Questions, and the recorded
Answers (with supporting Content). It outputs a plain-text report grouped by
species, then question, showing each answer alongside a short content snippet.
"""

from datetime import datetime
from typing import List, Tuple
from sqlalchemy import select, create_engine
from sqlalchemy.orm import Session
from itertools import groupby

from models import (
    DB_URI,
    Species,
    Question,
    Answer,
    Link,
    SpeciesQuestion,
    QuestionLink,
)

SNIPPET_LEN = 40

DB_ENGINE = create_engine(DB_URI)


def generate_species_qna_report(session: Session) -> str:
    """Build a flat text report of all answers.

    Each row contains a csv readable:
        [species],[question],[answer],[context snippet],[url]

    Args:
        session: SQLAlchemy ORM session.

    Returns:
        str: Formatted report with one line per (species, question, etc).
    """
    stmt = (
        select(
            Species.id.label("species_id"),
            Species.name.label("species_name"),
            Question.id.label("question_id"),
            Question.text.label("question_text"),
            Answer.id.label("answer_id"),
            Answer.answer_text.label("answer_text"),
            Answer.reason.label("reason"),
            Answer.evidence.label("evidence"),
            Answer.context.label("context"),
            Link.url.label("url"),
        )
        .join(SpeciesQuestion, SpeciesQuestion.species_id == Species.id)
        .join(Question, Question.id == SpeciesQuestion.question_id)
        .join(QuestionLink, QuestionLink.question_id == Question.id)
        .join(Answer, Answer.question_link_id == QuestionLink.id)
        .join(Link, Link.id == QuestionLink.link_id)
        .order_by("species_name", "question_id", "answer_id")
    )

    rows: List[Tuple] = session.execute(stmt).all()

    # species-question pairs with NO links at all
    no_link_stmt = (
        select(
            Species.id.label("species_id"),
            Species.name.label("species_name"),
            Question.id.label("question_id"),
            Question.text.label("question_text"),
        )
        .join(SpeciesQuestion, SpeciesQuestion.species_id == Species.id)
        .join(Question, Question.id == SpeciesQuestion.question_id)
        .outerjoin(QuestionLink, QuestionLink.question_id == Question.id)
        .where(QuestionLink.id.is_(None))
        .order_by("species_name", "question_id")
    )
    no_link_rows = session.execute(no_link_stmt).all()

    # collect species-question keys that already have at least one answer row
    have_answer_keys = {(r.species_id, r.question_id) for r in rows}

    lines: List[str] = []
    lines.append("species,question,answer,reason,evidence,url,context example")

    # answered rows
    for (species_name,), species_group in groupby(
        rows, key=lambda r: (r.species_name,)
    ):

        def qkey(r):
            return (r.question_id, r.question_text)

        for (_, question_text), q_group in groupby(species_group, key=qkey):
            for r in q_group:
                base_question_text = question_text.replace(
                    species_name, "[species]"
                )
                context = r.context
                quoted_index = context.lower().find(species_name.lower())
                if quoted_index != -1:
                    start = max(0, quoted_index - SNIPPET_LEN)
                    end = min(len(context), quoted_index + SNIPPET_LEN)
                    snippet = context[start:end].strip()
                    if start > 0:
                        snippet = "... " + snippet
                    if end < len(context):
                        snippet = snippet + " ..."
                    context_example = snippet
                else:
                    context_example = context[: SNIPPET_LEN * 2]

                lines.append(
                    f'"{species_name}","{base_question_text}","{r.answer_text}","{r.reason}","'
                    + " ... ".join(  # noqa: W503
                        e.replace("\n", " ").replace("\r", " ").strip()
                        for e in r.evidence
                    )
                    + f'","{r.url}","{context_example}"'  # noqa: W503
                )

    # add missing rows for species-question pairs with no links at all
    for r in no_link_rows:
        key = (r.species_id, r.question_id)
        if key in have_answer_keys:
            continue
        base_question_text = r.question_text.replace(
            r.species_name, "[species]"
        )
        lines.append(
            f'"{r.species_name}","{base_question_text}","none","no links found in search","n/a"'
        )

    return "\n".join(lines)


def write_species_qna_report(session: Session, path: str) -> None:
    """Generate and write the species-question-answer report to a file.

    Args:
        session: SQLAlchemy ORM session.
        path: Output file path.

    Returns:
        None
    """
    report = generate_species_qna_report(session)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(report)


if __name__ == "__main__":
    with Session(DB_ENGINE) as session:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"species_question_answer_report_{timestamp}.csv"
        write_species_qna_report(session, filename)
