"""SQLAlchemy model definitions for a normalized traits/question-answer pipeline (v2).

This module defines a normalized schema for:
  * Species
  * Questions (each includes its search keyword phrase)
  * A mapping of species <> questions (many-to-many)
  * Links (URLs) and fetched Content
  * Associations of questions <> links (which links are relevant to which questions)
  * Many answers per (question, link), with JSON-serializable evidence list

Key changes from prior version:
  * `Question` now stores both the question text and the keyword phrase used to search.
  * Replaced the previous SpeciesQuestion entity with a simple mapping table that pairs
    species_id with question_id (no additional scoping state).
  * Folded keyword queries into `Question` (removed `KeywordQuery`).
  * Simplified `QuestionLink` to associate a `question_id` directly to a `link_id`
    (removed `species_question_id`).
  * Removed `QuestionLinkDiscovery`.
  * Kept `Content` as its own first-class entity referenced by `Link`.
  * `Answer` links to a `question_id` and a `link_id`, and stores `evidence` as a JSON list.

Additional integrity:
  * `Answer` includes a composite foreign key referencing the unique pair
    (`question_id`, `link_id`) on `QuestionLink`, ensuring answers only exist for
    established question-link associations.
"""

from __future__ import annotations

from typing import List

from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

DB_URI = "sqlite:///data/search_content_index.db"


class BaseNorm(DeclarativeBase):
    """Declarative base for normalized models."""

    pass


class Content(BaseNorm):
    """Fetched or extracted document content.

    Attributes:
        id (int): Surrogate primary key.
        content_hash (str): Hash of the content for deduplication.
        text (str): Raw content text.
        is_valid (int | None): Optional status flag for validation/quality checks.
        links (list[Link]): Links that reference this content (1:N).
    """

    __tablename__ = "content"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    content_hash: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Keep as integer for SQLite simplicity (treat non-zero as True if desired).
    is_valid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    links: Mapped[List["Link"]] = relationship(
        back_populates="content",
        passive_deletes=True,
    )


class Link(BaseNorm):
    """A unique URL that may have associated fetched content.

    Attributes:
        id (int): Surrogate primary key.
        url (str): Canonical URL (unique).
        content_id (int | None): Optional FK to Content.
        content (Content | None): Related content instance.
        question_links (list[QuestionLink]): Associations of this link to questions.
        answers (list[Answer]): Convenience view of answers via question_links.
    """

    __tablename__ = "links"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    url: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    content_id: Mapped[int | None] = mapped_column(
        ForeignKey("content.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    fetch_error: Mapped[str | None] = mapped_column(String, nullable=True)

    content: Mapped[Content | None] = relationship(back_populates="links")

    question_links: Mapped[List["QuestionLink"]] = relationship(
        back_populates="link",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    answers: Mapped[List["Answer"]] = relationship(
        "Answer",
        back_populates="link",
        primaryjoin="Link.id == Answer.link_id",
        foreign_keys="Answer.link_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Species(BaseNorm):
    """A species record.

    Attributes:
        id (int): Surrogate primary key.
        name (str): Unique species name.
        questions (list[Question]): Related questions via the mapping table.
        species_questions (list[SpeciesQuestion]): Mapping rows to questions.
    """

    __tablename__ = "species"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )

    species_questions: Mapped[List["SpeciesQuestion"]] = relationship(
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    questions: Mapped[List["Question"]] = relationship(
        secondary="species_questions",
        primaryjoin="Species.id==SpeciesQuestion.species_id",
        secondaryjoin="Question.id==SpeciesQuestion.question_id",
        viewonly=True,
    )


class Question(BaseNorm):
    """A reusable question template that includes its search keyword phrase.

    Attributes:
        id (int): Surrogate primary key.
        text (str): Unique question text.
        keyword_phrase (str): Search phrase used to discover relevant links.
        species (list[Species]): Related species via the mapping table.
        species_questions (list[SpeciesQuestion]): Mapping rows to species.
        question_links (list[QuestionLink]): Links associated with this question.
        answers (list[Answer]): Answers extracted for this question across links.
    """

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    text: Mapped[str] = mapped_column(
        Text, nullable=False, unique=True, index=True
    )
    keyword_phrase: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )

    species_questions: Mapped[List["SpeciesQuestion"]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    species: Mapped[List["Species"]] = relationship(
        secondary="species_questions",
        primaryjoin="Question.id==SpeciesQuestion.question_id",
        secondaryjoin="Species.id==SpeciesQuestion.species_id",
        viewonly=True,
    )

    question_links: Mapped[List["QuestionLink"]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    answers: Mapped[List["Answer"]] = relationship(
        "Answer",
        back_populates="question",
        primaryjoin="Question.id == Answer.question_id",
        foreign_keys="Answer.question_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SpeciesQuestion(BaseNorm):
    """Mapping between a species and a question (many-to-many).

    This table contains no additional scoping state; it simply relates a species to a question.

    Attributes:
        species_id (int): FK to Species.
        question_id (int): FK to Question.
    """

    __tablename__ = "species_questions"

    species_id: Mapped[int] = mapped_column(
        ForeignKey("species.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "species_id", "question_id", name="uq_species_question"
        ),
        Index("ix_species_questions_spp_q", "species_id", "question_id"),
    )

    species: Mapped[Species] = relationship(back_populates="species_questions")
    question: Mapped[Question] = relationship(
        back_populates="species_questions"
    )


class QuestionLink(BaseNorm):
    """Association of a Question to a Link.

    A link can be reused across multiple questions; this table records which links
    are relevant to which questions.

    Attributes:
        id (int): Surrogate primary key.
        question_id (int): FK to Question.
        link_id (int): FK to Link.
        question (Question): Related question.
        link (Link): Related link.
        answers (list[Answer]): Answers derived for this (question, link) pair.
    """

    __tablename__ = "question_links"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    link_id: Mapped[int] = mapped_column(
        ForeignKey("links.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        UniqueConstraint("question_id", "link_id", name="uq_question_link"),
        Index("ix_question_links_q_l", "question_id", "link_id"),
    )

    question: Mapped[Question] = relationship(back_populates="question_links")
    link: Mapped[Link] = relationship(back_populates="question_links")

    answers: Mapped[List["Answer"]] = relationship(
        "Answer",
        back_populates="question_link",
        primaryjoin="QuestionLink.id == Answer.question_link_id",
        foreign_keys="Answer.question_link_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Answer(BaseNorm):
    """An extracted answer for a specific (question, link) pair.

    Attributes:
        id (int): Surrogate primary key.
        question_link_id (int): FK to QuestionLink for fast joins and cascading.
        question_id (int): Redundant FK to Question for query ergonomics.
        link_id (int): Redundant FK to Link for query ergonomics.
        answer_text (str): Extracted answer text.
        reason (str): Reason/label for answer state (e.g., 'supported', 'not_enough_information').
        evidence (list): JSON-serializable list of evidence snippets/objects.

    Notes:
        A composite foreign key enforces that (question_id, link_id) exists on QuestionLink.
        Redundant columns (question_id, link_id) speed up common filters and group-bys.
    """

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    # Direct handle to the association row (for cascades & joins)
    question_link_id: Mapped[int] = mapped_column(
        ForeignKey("question_links.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Redundant, but ergonomic for filtering and reporting
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    link_id: Mapped[int] = mapped_column(
        ForeignKey("links.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    context: Mapped[str] = mapped_column(String, nullable=False)

    # Enforce that (question_id, link_id) exists in QuestionLink.uq_question_link
    __table_args__ = (
        ForeignKeyConstraint(
            ["question_id", "link_id"],
            ["question_links.question_id", "question_links.link_id"],
            name="fk_answers_question_link_pair",
            ondelete="RESTRICT",
        ),
        Index("ix_answers_q_l", "question_id", "link_id"),
    )

    question_link: Mapped["QuestionLink"] = relationship(
        "QuestionLink",
        back_populates="answers",
        primaryjoin="Answer.question_link_id == QuestionLink.id",
        foreign_keys="Answer.question_link_id",
    )
    question: Mapped["Question"] = relationship(
        "Question",
        back_populates="answers",
        primaryjoin="Answer.question_id == Question.id",
        foreign_keys="Answer.question_id",
    )

    link: Mapped["Link"] = relationship(
        "Link",
        back_populates="answers",
        primaryjoin="Answer.link_id == Link.id",
        foreign_keys="Answer.link_id",
    )
