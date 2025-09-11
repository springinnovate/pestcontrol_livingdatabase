"""Script to report questions that have no links associted with them."""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from models import (
    Species,
    Question,
    SpeciesQuestion,
    QuestionLink,
    DB_URI,
)


def iter_species_questions_without_links(
    session: Session,
) -> Iterable[tuple[str, str]]:  # noqa: D208
    """Loops through all the questions without links associted.

    returns a (species_name, question_text) iterator
    """
    stmt = (
        select(Species.name, Question.text)
        .join(SpeciesQuestion, SpeciesQuestion.species_id == Species.id)
        .join(Question, Question.id == SpeciesQuestion.question_id)
        .outerjoin(QuestionLink, QuestionLink.question_id == Question.id)
        .group_by(Species.id, Species.name, Question.id, Question.text)
        .having(func.count(QuestionLink.id) == 0)
        .order_by(Species.name.asc(), Question.text.asc())
    )
    yield from session.execute(stmt).all()


def main() -> None:
    """Entry point."""
    engine = create_engine(DB_URI)
    out: dict[str, list[str]] = defaultdict(list)
    with Session(engine) as session:
        for species_name, question_text in iter_species_questions_without_links(
            session
        ):
            out[species_name].append(question_text)

    # Print in requested format
    for species_name in sorted(out.keys()):
        print(f"[{species_name}]")
        for q in out[species_name]:
            print(f"\t[{q}]")


if __name__ == "__main__":
    main()
