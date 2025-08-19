from typing import List
from sqlalchemy import create_engine, String, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.schema import Index


DB_URI = "sqlite:///data/search_results.db"


class Base(DeclarativeBase):
    pass


class SearchResult(Base):
    __tablename__ = "search_results"

    # Primary key: the full English question (unique per DB)
    question: Mapped[str] = mapped_column(String, primary_key=True)

    # Name of species (from predator_list, pest_list, or full_species_list)
    species_name: Mapped[str] = mapped_column(String, nullable=False)

    # The exact keyword-based search string sent to the engine
    keyword_query: Mapped[str] = mapped_column(String, nullable=False)

    # JSON list of links returned by the search
    links: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)

    __table_args__ = (
        Index("ix_species_name", "species_name"),
        Index("ix_keyword_query", "keyword_query"),
    )


def get_session(
    sqlite_path: str = DB_URI,
) -> Session:
    engine = create_engine(sqlite_path, future=True)
    Base.metadata.create_all(engine)
    return Session(engine)
