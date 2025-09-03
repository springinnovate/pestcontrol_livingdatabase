"""SQLAlchemy model definitions for trait pipeline."""

from typing import List
from sqlalchemy import String

from sqlalchemy import (
    ForeignKey,
    Integer,
    JSON,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

DB_URI = "sqlite:///data/search_content_index.db"


class BaseNorm(DeclarativeBase):
    pass


class Content(BaseNorm):
    __tablename__ = "content"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    content_hash: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    links: Mapped[List["Link"]] = relationship(
        back_populates="content", cascade="all, delete"
    )


class Link(BaseNorm):
    __tablename__ = "links"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    url: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    content_id: Mapped[int | None] = mapped_column(
        ForeignKey("content.id", ondelete="SET NULL"), nullable=True
    )

    content: Mapped[Content | None] = relationship(back_populates="links")
    search_heads: Mapped[List["SearchHead"]] = relationship(
        secondary="search_result_links",
        back_populates="links",
    )
    answers: Mapped[List["Answer"]] = relationship(back_populates="link")


class Question(BaseNorm):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    text: Mapped[str] = mapped_column(
        Text, nullable=False, unique=True, index=True
    )

    search_head: Mapped["SearchHead"] = relationship(
        back_populates="question", uselist=False, cascade="all, delete"
    )
    question_keyword: Mapped["QuestionKeyword"] = relationship(
        back_populates="question", uselist=False, cascade="all, delete"
    )


class Species(BaseNorm):
    __tablename__ = "species"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )


class KeywordQuery(BaseNorm):
    __tablename__ = "keyword_queries"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    query: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )

    question_keywords: Mapped[List["QuestionKeyword"]] = relationship(
        back_populates="keyword_query"
    )
    answers: Mapped[List["Answer"]] = relationship(
        back_populates="keyword_query"
    )


class QuestionKeyword(BaseNorm):
    __tablename__ = "question_keywords"
    # 1:1 mapping of question -> keyword query (query is tied directly to the question)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    keyword_query_id: Mapped[int] = mapped_column(
        ForeignKey("keyword_queries.id", ondelete="RESTRICT"),
        nullable=False,
    )

    question: Mapped[Question] = relationship(back_populates="question_keyword")
    keyword_query: Mapped[KeywordQuery] = relationship(
        back_populates="question_keywords"
    )


class SearchHead(BaseNorm):
    __tablename__ = "search_heads"
    # primary key is the question id; keyword query relation moved to QuestionKeyword
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    species_id: Mapped[int] = mapped_column(
        ForeignKey("species.id", ondelete="RESTRICT"), nullable=False
    )

    question: Mapped[Question] = relationship(back_populates="search_head")
    species: Mapped[Species] = relationship()

    links: Mapped[List[Link]] = relationship(
        secondary="search_result_links",
        back_populates="search_heads",
    )


class SearchResultLink(BaseNorm):
    __tablename__ = "search_result_links"
    search_head_id: Mapped[int] = mapped_column(
        ForeignKey("search_heads.question_id", ondelete="CASCADE"),
        primary_key=True,
    )
    link_id: Mapped[int] = mapped_column(
        ForeignKey("links.id", ondelete="CASCADE"),
        primary_key=True,
    )
    __table_args__ = (
        UniqueConstraint(
            "search_head_id", "link_id", name="uq_search_head_link_id"
        ),
    )


class Answer(BaseNorm):
    __tablename__ = "answers"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    keyword_query_id: Mapped[int] = mapped_column(
        ForeignKey("keyword_queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    link_id: Mapped[int] = mapped_column(
        ForeignKey("links.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # extracted result
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(
        String, nullable=False, default="not_enough_information", index=True
    )
    evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    keyword_query: Mapped[KeywordQuery] = relationship(back_populates="answers")
    link: Mapped[Link] = relationship(back_populates="answers")
