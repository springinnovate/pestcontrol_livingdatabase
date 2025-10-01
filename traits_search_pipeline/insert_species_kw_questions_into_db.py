"""Oneshot to insert raw species/questions/keyword search into the database."""

import argparse
from pathlib import Path

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
    """Parse input lines and insert species/questions/mappings; with --dry_run, list would-be inserts only."""
    ap = argparse.ArgumentParser(
        description="Insert [species]:[keyword query]:[question] lines into the database."
    )
    ap.add_argument("infile", type=Path, help="input file path")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="List what would be inserted (species, questions, mappings) without modifying the database.",
    )
    args = ap.parse_args()

    DB_ENGINE = create_engine(DB_URI)
    BaseNorm.metadata.create_all(DB_ENGINE)
    inserted = updated = skipped = 0

    # collections for dry run preview
    would_create_species = set()
    would_create_questions = (
        []
    )  # tuples: (species_name, keyword_query, question_text)
    would_create_mappings = []  # tuples: (species_name, question_text)

    with Session(DB_ENGINE) as sess, args.infile.open(
        "r", encoding="utf-8"
    ) as f:
        if args.dry_run:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    species_name, keyword_query, question_text = line.split(
                        ":", 2
                    )
                except Exception:
                    print(f"error with {line}")
                    raise

                species_row = sess.scalar(
                    select(Species).where(Species.name == species_name)
                )
                if species_row is None:
                    would_create_species.add(species_name)

                question_row = sess.scalar(
                    select(Question).where(Question.text == question_text)
                )
                if question_row is None:
                    would_create_questions.append(
                        (species_name, keyword_query, question_text)
                    )
                    # mapping would also be new if species exists or will exist
                    would_create_mappings.append((species_name, question_text))
                else:
                    # question exists; check mapping
                    if species_row is None:
                        # species will be created, so mapping will be new as well
                        would_create_mappings.append(
                            (species_name, question_text)
                        )
                    else:
                        mapping_exists = sess.scalar(
                            select(SpeciesQuestion).where(
                                SpeciesQuestion.species_id == species_row.id,
                                SpeciesQuestion.question_id == question_row.id,
                            )
                        )
                        if not mapping_exists:
                            would_create_mappings.append(
                                (species_name, question_text)
                            )

            # output preview
            print("=== DRY RUN: WOULD CREATE SPECIES ===")
            for s in sorted(would_create_species):
                print(f"- {s}")

            print("\n=== DRY RUN: WOULD CREATE QUESTIONS ===")
            # show unique question creations (question_text uniqueness governs creation)
            seen_q = set()
            for (
                species_name,
                keyword_query,
                question_text,
            ) in would_create_questions:
                if question_text in seen_q:
                    continue
                seen_q.add(question_text)
                print(f"- [{keyword_query}] {question_text}")

            print(
                "\n=== DRY RUN: WOULD CREATE MAPPINGS (Species ↔ Question) ==="
            )
            for species_name, question_text in sorted(
                set(would_create_mappings)
            ):
                print(f"- {species_name} ↔ {question_text}")

            print("\n=== DRY RUN: COUNTS ===")
            print(f"species_to_create: {len(would_create_species)}")
            print(f"questions_to_create: {len(seen_q)}")
            print(f"mappings_to_create: {len(set(would_create_mappings))}")
            return

        # live insert path
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                species_name, keyword_query, question_text = line.split(":", 2)
            except Exception:
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
                # updated if keyword changed, else skipped (optional granularity)
                if before_q.keyword_phrase != q.keyword_phrase:
                    updated += 1
                else:
                    skipped += 1

        sess.commit()

    print(f"inserted: {inserted}, updated: {updated}, skipped: {skipped}")


if __name__ == "__main__":
    main()
