"""
Generate a species–question–answer report from the database.

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
    Content,
    SpeciesQuestion,
    QuestionLink,
)

SNIPPET_LEN = 40

DB_ENGINE = create_engine(DB_URI)


def _snippet(text: str, n: int = SNIPPET_LEN) -> str:
    return (
        (text[:n].rstrip() + "...") if text and len(text) > n else (text or "")
    )


def generate_species_qna_report(session: Session) -> str:
    """
    Build a text report grouped as:
    [species]
        [question]
           [answer] [context]
           …

    Args:
        session: SQLAlchemy ORM session.

    Returns:
        str: Formatted report.
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
            Link.url.label("url"),
            Content.text.label("content_text"),
        )
        .join(SpeciesQuestion, SpeciesQuestion.species_id == Species.id)
        .join(Question, Question.id == SpeciesQuestion.question_id)
        .join(QuestionLink, QuestionLink.question_id == Question.id)
        .join(Answer, Answer.question_link_id == QuestionLink.id)
        .join(Link, Link.id == QuestionLink.link_id)
        .join(Content, Content.id == Link.content_id)
        .order_by("species_name", "question_id", "answer_id")
    )

    rows: List[Tuple] = session.execute(stmt).all()

    lines: List[str] = []
    lines.append("species,question,answer,reason,url")
    for (species_name,), species_group in groupby(
        rows, key=lambda r: (r.species_name,)
    ):
        # lines.append(f"[{species_name}]")

        # group by question within species
        def qkey(r):
            return (r.question_id, r.question_text)

        for (_, question_text), q_group in groupby(species_group, key=qkey):
            for r in q_group:
                # ctx = _snippet(r.content_text)
                lines.append(
                    f'"{species_name}","{question_text}","{r.answer_text}","{r.reason}","{r.url}"'
                )
    return "\n".join(lines)


def write_species_qna_report(session: Session, path: str) -> None:
    """
    Generate and write the species-question-answer report to a file.

    Args:
        session: SQLAlchemy ORM session.
        path: Output file path.

    Returns:
        None
    """
    report = generate_species_qna_report(session)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    with Session(DB_ENGINE) as session:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"species_question_answer_report_{timestamp}.txt"
        write_species_qna_report(session, filename)
