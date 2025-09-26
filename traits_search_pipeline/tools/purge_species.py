"""Purge a species and its related links/answers etc from the db.

run at the command line like this: python -m tools.purge_species species_name
"""

from pathlib import Path
import argparse
import datetime
import shutil

from sqlalchemy import create_engine, select, delete, exists
from sqlalchemy.orm import Session

from models import (
    DB_URI,
    Species,
    SpeciesQuestion,
    Question,
    QuestionLink,
    Answer,
)


DB_ENGINE = create_engine(DB_URI)


def delete_species_cascade(
    session: Session, species_name: str, dry_run: bool = False
) -> dict:
    """# noqa: D205, D210, D415
    Delete a species and all downstream records tied to it:
      - Answers for questions linked to the species
      - QuestionLinks for those questions
      - SpeciesQuestion mappings for that species
      - Orphan Questions (only those no longer linked to any species)
      - Global scrub: any Questions with no SpeciesQuestion and no QuestionLink
      - Finally the Species row itself
    Leaves Link records intact.

    If dry_run=True, no rows are deleted; instead, ids and counts of rows
    that would be deleted are returned in 'preview'.
    """
    stats = {
        "answers_deleted": 0,
        "question_links_deleted": 0,
        "species_questions_deleted": 0,
        "questions_deleted": 0,
        "species_deleted": 0,
        "preview": {
            "answer_ids": [],
            "question_link_ids": [],
            "species_question_pairs": [],  # (species_id, question_id)
            "question_ids": [],  # species-scoped orphan questions
            "global_orphan_question_ids": [],  # global scrub pass
            "species_ids": [],
        },
    }

    try:
        with session.begin():
            species_select = session.execute(
                select(Species).where(Species.name == species_name)
            ).scalar_one_or_none()
            if species_select is None:
                return stats

            question_ids_subquery = (
                select(SpeciesQuestion.question_id)
                .where(SpeciesQuestion.species_id == species_select.id)
                .subquery()
            )

            del_answers = session.execute(
                delete(Answer)
                .where(
                    Answer.question_id.in_(
                        select(question_ids_subquery.c.question_id)
                    )
                )
                .returning(Answer.id)
            )
            stats["answers_deleted"] = len(del_answers.fetchall())

            del_q_links = session.execute(
                delete(QuestionLink)
                .where(
                    QuestionLink.question_id.in_(
                        select(question_ids_subquery.c.question_id)
                    )
                )
                .returning(QuestionLink.id)
            )
            stats["question_links_deleted"] = len(del_q_links.fetchall())

            delete_speciesquestion = session.execute(
                delete(SpeciesQuestion)
                .where(SpeciesQuestion.species_id == species_select.id)
                .returning(
                    SpeciesQuestion.species_id, SpeciesQuestion.question_id
                )
            )
            stats["species_questions_deleted"] = len(
                delete_speciesquestion.fetchall()
            )

            orphan_question_subquery = (
                select(Question.id)
                .where(
                    Question.id.in_(
                        select(question_ids_subquery.c.question_id)
                    ),
                    ~exists().where(SpeciesQuestion.question_id == Question.id),
                )
                .subquery()
            )
            del_questions = session.execute(
                delete(Question)
                .where(Question.id.in_(select(orphan_question_subquery.c.id)))
                .returning(Question.id)
            )
            species_orphan_deleted = len(del_questions.fetchall())
            stats["questions_deleted"] += species_orphan_deleted

            global_orphan_question_subquery = (
                select(Question.id)
                .where(
                    ~exists().where(SpeciesQuestion.question_id == Question.id),
                    ~exists().where(QuestionLink.question_id == Question.id),
                )
                .subquery()
            )
            del_global_questions = session.execute(
                delete(Question)
                .where(
                    Question.id.in_(
                        select(global_orphan_question_subquery.c.id)
                    )
                )
                .returning(Question.id)
            )
            global_orphan_deleted = len(del_global_questions.fetchall())
            stats["questions_deleted"] += global_orphan_deleted

            del_species = session.execute(
                delete(Species)
                .where(Species.id == species_select.id)
                .returning(Species.id)
            )
            stats["species_deleted"] = len(del_species.fetchall())

            if dry_run:
                session.rollback()
            else:
                session.commit()
            return stats
    except Exception:
        session.rollback()
        raise


def main():
    """CLI entry point for deleting a species and related records from the database.

    By default, runs in dry-run mode to show what would be deleted. Use
    --not_dry_run to perform the actual deletion (after making a backup).
    """
    parser = argparse.ArgumentParser(
        description="Delete a species and downstream records from the database."
    )
    parser.add_argument(
        "species_name", help="The exact name of the species to delete."
    )
    parser.add_argument(
        "--not_dry_run",
        action="store_true",
        help="Actually perform the delete. By default, the script runs in dry-run mode.",
    )
    args = parser.parse_args()

    dry_run = not args.not_dry_run
    if not dry_run:
        # file-copy backup
        _db_path_str = DB_URI.replace("sqlite:////", "/").replace(
            "sqlite:///", ""
        )
        _db_path = Path(_db_path_str).resolve()

        _backup_path = _db_path.with_suffix(
            _db_path.suffix + f".{datetime.datetime.now():%Y%m%d_%H%M%S}.bak"
        )
        shutil.copy2(_db_path, _backup_path)
        print(f"Backup written to {_backup_path}")

    with Session(DB_ENGINE) as session:
        result = delete_species_cascade(
            session, species_name=args.species_name, dry_run=dry_run
        )
        print(result)


if __name__ == "__main__":
    main()
