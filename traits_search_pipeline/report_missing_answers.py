"""Script to report questions that have no answers associted with them."""

from __future__ import annotations
from typing import Iterable

from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import Session

from models import (
    Question,
    QuestionLink,
    DB_URI,
    Content,
    Answer,
    Link,
)


def iter_questions_without_answers(
    session: Session,
) -> Iterable[tuple[str, str]]:  # noqa: D208
    """Loops through all the questions without links associted.

    returns a (species_name, question_text) iterator
    """
    stmt = (
        select(QuestionLink, Question, Link, Content, Answer)
        .join(Question, QuestionLink.question)
        .join(Link, QuestionLink.link)
        .join(Content, Link.content)
        .outerjoin(
            Answer,
            and_(
                Answer.question_id == QuestionLink.question_id,
                Answer.link_id == QuestionLink.link_id,
            ),
        )
        .where(and_(Answer.id.is_(None), Content.is_valid == 1))
        .execution_options(stream_results=True, yield_per=200)
    )
    yield from session.execute(stmt).all()


def main() -> None:
    """Entry point."""
    engine = create_engine(DB_URI)
    with Session(engine) as session:
        with open("questions_to-answer.txt", "w") as file:
            file.write("{question.id},{link.id},{content.is_valid},{answer}\n")
            for (
                question_link,
                question,
                link,
                content,
                answer,
            ) in iter_questions_without_answers(session):
                file.write(
                    f"{question.id},{link.id},{content.is_valid},{answer}\n"
                )


if __name__ == "__main__":
    main()
