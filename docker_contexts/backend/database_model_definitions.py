"""Database definitions for news articles and their classifications.

Defined from: https://docs.google.com/spreadsheets/d/1yZwc7fPB0kHI9F5jdgUKuNflgkQF7SHS/edit#gid=1487741928
"""
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from typing import List
from typing import Optional

RESERVED_NAMES = ['covariate']


class Base(DeclarativeBase):
    pass


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[str]
    data_contributor: Mapped[str]
    data_contributor_contact_info: Mapped[str]
    study_metadata: Mapped[str]
    response_types: Mapped[str]
    samples: Mapped[Optional["Sample"]] = relationship(back_populates="study")
    paper_dois = relationship(
        "DOI", secondary="StudyDOIAssociation", back_populates="studies")


class DOI(Base):
    __tablename__ = 'doi'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[str]
    studies = relationship(
        "Study", secondary="StudyDOIAssociation", back_populates="paper_dois")


class StudyDOIAssociation(Base):
    __tablename__ = 'study_doi'
    study_id: Mapped[int] = mapped_column(
        ForeignKey('study.id_key'), primary_key=True)
    doi_id: Mapped[int] = mapped_column(
        ForeignKey('doi.id_key'), primary_key=True)


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study: Mapped[Study] = relationship(back_populates="id_key")
    latitude: Mapped[float]
    longitude: Mapped[float]
    manager: Mapped[Optional[str]]
    year: Mapped[int]
    month: Mapped[Optional[int]]
    day: Mapped[Optional[int]]
    local_time: Mapped[Optional[str]]
    replicate: Mapped[Optional[str]]
    sampling_effort: Mapped[str]
    observation: Mapped[str]
    observer_id: Mapped[str]
    response_variable: Mapped[str]
    units: Mapped[str]
    sampling_method: Mapped[str]
    sampler_type: Mapped[str]
    functional_type: Mapped[str]
    crop_commercial_name: Mapped[str]
    crop_latin_name: Mapped[str]

    abundance_class: Mapped[Optional[str]]
    order: Mapped[Optional[str]]
    family: Mapped[Optional[str]]
    genus: Mapped[Optional[str]]
    species: Mapped[Optional[str]]
    sub_species: Mapped[Optional[str]]
    morphospecies: Mapped[Optional[str]]
    life_stage: Mapped[Optional[str]]
    pest_class: Mapped[Optional[str]]
    pest_order: Mapped[Optional[str]]
    pest_family: Mapped[Optional[str]]
    pest_species: Mapped[Optional[str]]
    pest_sub_species: Mapped[Optional[str]]
    pest_morphospecies: Mapped[Optional[str]]
    pest_life_stage: Mapped[Optional[str]]
    enemy_class: Mapped[Optional[str]]
    enemy_order: Mapped[Optional[str]]
    enemy_family: Mapped[Optional[str]]
    enemy_species: Mapped[Optional[str]]
    enemy_sub_species: Mapped[Optional[str]]
    enemy_morphospecies: Mapped[Optional[str]]
    enemy_lifestage: Mapped[Optional[str]]
    growth_stage_of_crop_at_sampling: Mapped[Optional[str]]
    covariates: Mapped[List["Covariate"]] = relationship(
        back_populates="sample")


class Covariate(Base):
    __tablename__ = 'covariate'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    covariate_category: Mapped[Optional[str]]
    covariate_name: Mapped[str]
    covariate_value: Mapped[str]
    sample: Mapped["Sample"] = relationship(back_populates="covariates")
