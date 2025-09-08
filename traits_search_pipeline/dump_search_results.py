from collections import defaultdict
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from models import Species, Question, SearchHead, DB_URI


def questions_without_links_by_species(
    engine_url: str = DB_URI,
) -> dict[str, list[str]]:
    engine = create_engine(engine_url, future=True)
    out: dict[str, list[str]] = defaultdict(list)
    with Session(engine) as sess:
        stmt = (
            select(Species.name, Question.text)
            .join(SearchHead, SearchHead.species_id == Species.id)
            .join(Question, Question.id == SearchHead.question_id)
            .where(~SearchHead.links.any())
            .order_by(Species.name.asc(), Question.text.asc())
        )
        for species_name, question_text in sess.execute(stmt):
            out[species_name].append(question_text)
    return dict(out)


if __name__ == "__main__":
    grouped = questions_without_links_by_species()
    for species_name, questions in grouped.items():
        print(species_name)
        # for q in questions:
        #     print(f"  - {q}")
