"""Oneshot to insert raw species/questions/keyword search into the database."""

import argparse
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import Species, Question, SpeciesQuestion, BaseNorm, DB_URI


def get_or_create_species(sess: Session, name: str) -> Species:
    obj = sess.scalar(select(Species).where(Species.name == name))
    if obj:
        return obj
    obj = Species(name=name)
    sess.add(obj)
    sess.flush()
    return obj


def get_or_create_question(
    sess: Session, text: str, keyword_phrase: str
) -> Question:
    obj = sess.scalar(select(Question).where(Question.text == text))
    if obj:
        if obj.keyword_phrase != keyword_phrase:
            obj.keyword_phrase = keyword_phrase
        return obj
    obj = Question(text=text, keyword_phrase=keyword_phrase)
    sess.add(obj)
    sess.flush()
    return obj


def ensure_species_question(
    sess: Session, species_id: int, question_id: int
) -> None:
    exists = sess.scalar(
        select(SpeciesQuestion).where(
            SpeciesQuestion.species_id == species_id,
            SpeciesQuestion.question_id == question_id,
        )
    )
    if not exists:
        sess.add(
            SpeciesQuestion(species_id=species_id, question_id=question_id)
        )


def main():
    ap = argparse.ArgumentParser(
        description="Insert [species]:[keyword query]:[question] lines into the database."
    )
    ap.add_argument("infile", type=Path, help="input file path")
    args = ap.parse_args()

    DB_ENGINE = create_engine(DB_URI)
    BaseNorm.metadata.create_all(DB_ENGINE)
    inserted = updated = skipped = 0

    with Session(DB_ENGINE) as sess, args.infile.open(
        "r", encoding="utf-8"
    ) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                species_name, keyword_query, question_text = line.split(":")
            except:
                print(f"error with {line}")
                raise

            spp = get_or_create_species(sess, species_name)
            before_q = sess.scalar(
                select(Question).where(Question.text == question_text)
            )
            q = get_or_create_question(sess, question_text, keyword_query)
            ensure_species_question(sess, spp.id, q.id)

            if before_q is None:
                inserted += 1
            else:
                updated += 1

        sess.commit()

    print(f"inserted: {inserted}, updated: {updated}, skipped: {skipped}")


if __name__ == "__main__":
    main()
