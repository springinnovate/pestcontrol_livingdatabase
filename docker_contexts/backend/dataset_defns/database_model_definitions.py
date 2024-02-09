"""Database definitions for news articles and their classifications.

Defined from: https://docs.google.com/spreadsheets/d/1yZwc7fPB0kHI9F5jdgUKuNflgkQF7SHS/edit#gid=1487741928
"""
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List
from typing import Optional


class Base(DeclarativeBase):
    pass


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id = Mapped[str]
    data_contributor = Mapped[str]
    data_contributor_contact_info = Mapped[str]
    paper_doi = Mapped[List[str]]
    metadata = Mapped[str]
    response_types = Mapped[str]
    names_of_covariates = Mapped[List[str]]
    samples: Mapped[Optional["Sample"]] = relationship(back_populates="study")


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study: Mapped[Study]
    latitude = Mapped[float]
    longitude = Mapped[float]
    manager = Mapped[Optional[str]]
    year = Mapped[int]
    month = Mapped[Optional[int]]
    day = Mapped[Optional[int]]
    local_time = Mapped[Optional[str]]
    replicate = Mapped[Optional[str]]
    sampling_effort = Mapped[str]
    observation = Mapped[str]
    observer_id = Mapped[str]
    response_variable = Mapped[str]
    units = Mapped[str]
    sampling_method = Mapped[str]
    sampler_type = Mapped[str]
    functional_type = Mapped[str]
    crop_commercial_name = Mapped[str]
    crop_latin_name = Mapped[str]
    growth_stage_of_crop_at_sampling = Mapped[Optional[str]]
    covariate_list = Mapped[Optional["Covariate"]] = relationship(
        back_populates="sample")


class Covariate(Base):
    __tablename__ = 'covariate'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str]
    name: Mapped[str]
    value: Mapped[str]
    sample: Mapped[Sample]


class Abudance(Base):
    __tablename__ = 'abudance'
    abundance_class: Mapped[Optional[str]]
    order: Mapped[Optional[str]]
    family: Mapped[Optional[str]]
    genus: Mapped[Optional[str]]
    species: Mapped[Optional[str]]
    sub_species: Mapped[Optional[str]]
    morphospecies: Mapped[Optional[str]]
    life_stage: Mapped[Optional[str]]


class Activity(Base):
    Pest_class: Mapped[Optional[str]]
    Pest_order: Mapped[Optional[str]]
    Pest_family: Mapped[Optional[str]]
    Pest_species: Mapped[Optional[str]]
    Pest_sub_species: Mapped[Optional[str]]
    Pest_morphospecies: Mapped[Optional[str]]
    Pest_life_stage: Mapped[Optional[str]]
    Enemy_class: Mapped[Optional[str]]
    Enemy_order: Mapped[Optional[str]]
    Enemy_family: Mapped[Optional[str]]
    Enemy_species: Mapped[Optional[str]]
    Enemy_sub_species: Mapped[Optional[str]]
    Enemy_morphospecies: Mapped[Optional[str]]
    Enemy_lifestage: Mapped[Optional[str]]
