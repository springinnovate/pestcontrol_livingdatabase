"""Database definitions for news articles and their classifications.

Defined from:
https://docs.google.com/spreadsheets/d/1yZwc7fPB0kHI9F5jdgUKuNflgkQF7SHS/edit#gid=1487741928
"""
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from typing import List
from typing import Optional
from sqlalchemy import Table, Column, Integer, JSON


COVARIATE_ID = 'covariate'
DOI_ID = 'doi'
RESERVED_NAMES = [COVARIATE_ID]

RESPONSE_TYPES = [
    'abundance',
    'activity',
    'production',
]

COORDINATE_PRECISION_FIELD = 'Desired coordinate precision'
COORDINATE_PRECISION_FULL_PRECISION_VALUE = 'full precision'
COORDINATE_PRECISION = [
    COORDINATE_PRECISION_FULL_PRECISION_VALUE,
    'round to whole number',
    '1 decimal place',
    '2 decmial places',
    '3 decimal places',
]

STUDY_ID = 'Study_ID'

STUDY_LEVEL_VARIABLES = [
    STUDY_ID,
    'Data contributor',
    'Data contributor contact info',
    'Paper(s) DOI',
    'Metadata',
    ('Response types', RESPONSE_TYPES),
    (COORDINATE_PRECISION_FIELD, COORDINATE_PRECISION),
]

MANAGER = 'Manager'
YEAR = 'Year'
MONTH = 'Month'
DAY = 'Day'
TIME = 'Time'
REPLICATE = 'Replicate'
SAMPLING_EFFORT = 'Sampling effort'
OBSERVATION = 'Observation'
OBSERVER_ID = 'Observer ID'
RESPONSE_VARIABLE = 'Response variable'
UNITS = 'Units'
SAMPLING_METHOD = 'Sampling method'
SAMPLER_TYPE = 'Sampler type'
FUNCTIONAL_TYPE = 'Functional type'
CROP_COMMERCIAL_NAME = 'Crop commercial name'
CROP_LATIN_NAME = 'Crop latin name'
GROWTH_STAGE_OF_CROP_AT_SAMPLING = 'Growth stage of crop at sampling'

BASE_FIELDS = [
    'Latitude',
    'Longitude',
    MANAGER,
    YEAR,
    MONTH,
    DAY,
    TIME,
    REPLICATE,
    SAMPLING_EFFORT,
    OBSERVATION,
    OBSERVER_ID,
    RESPONSE_VARIABLE,
    UNITS,
    SAMPLING_METHOD,
    SAMPLER_TYPE,
    FUNCTIONAL_TYPE,
    CROP_COMMERCIAL_NAME,
    CROP_LATIN_NAME,
    GROWTH_STAGE_OF_CROP_AT_SAMPLING,
    ]

FILTERABLE_FIELDS = [
    MANAGER,
    YEAR,
    REPLICATE,
    SAMPLING_EFFORT,
    OBSERVATION,
    OBSERVER_ID,
    RESPONSE_VARIABLE,
    UNITS,
    SAMPLING_METHOD,
    SAMPLER_TYPE,
    FUNCTIONAL_TYPE,
    CROP_COMMERCIAL_NAME,
    CROP_LATIN_NAME,
    GROWTH_STAGE_OF_CROP_AT_SAMPLING,
]

FIELDS_BY_REPONSE_TYPE = {
    'abundance': [
        'Class',
        'Order',
        'Family',
        'Genus ',
        'Species',
        'Sub-species',
        'Morphospecies',
        'Life stage',
        ],
    'activity': [
        'Pest class',
        'Pest order',
        'Pest family',
        'Pest species',
        'Pest sub-species',
        'Pest morphospecies',
        'Pest life stage',
        'Enemy class',
        'Enemy order',
        'Enemy family',
        'Enemy species',
        'Enemy sub-species',
        'Enemy morphospecies',
        'Enemy lifestage',
        ],
    'production': [],
}

if len(RESPONSE_TYPES) != len(set(RESPONSE_TYPES).union(set(FIELDS_BY_REPONSE_TYPE))):
    raise ValueError(
        'Response types and fields by response types do not have the same '
        'fields in database_model_definitions.py, open that file and find out '
        'what is going on.')


class Base(DeclarativeBase):
    pass


StudyDOIAssociation = Table(
    'study_doi_association', Base.metadata,
    Column('study_id', Integer, ForeignKey('study.id_key'), primary_key=True),
    Column('doi_id', Integer, ForeignKey('doi.id_key'), primary_key=True)
)


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[str]
    study_metadata: Mapped[Optional[str]]
    paper_dois = relationship(
        "DOI", secondary=StudyDOIAssociation, back_populates="studies")
    samples: Mapped[List["Sample"]] = relationship("Sample", back_populates="study")


class DOI(Base):
    __tablename__ = 'doi'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[str]
    studies = relationship(
        "Study", secondary=StudyDOIAssociation, back_populates="paper_dois")


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey('study.id_key'))
    study: Mapped[Study] = relationship("Study", back_populates="samples")
    latitude: Mapped[float]
    longitude: Mapped[float]
    manager: Mapped[Optional[str]]
    year: Mapped[int]
    month: Mapped[Optional[int]]
    day: Mapped[Optional[int]]
    local_time: Mapped[Optional[str]]
    replicate: Mapped[Optional[str]]
    sampling_effort: Mapped[Optional[str]]
    observation: Mapped[str]
    observer_id: Mapped[Optional[str]]
    response_type: Mapped[str]
    units: Mapped[Optional[str]]
    sampling_method: Mapped[str]
    sampler_type: Mapped[Optional[str]]
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
    sample_id: Mapped[int] = mapped_column(ForeignKey('sample.id_key'))
    covariate_category: Mapped[Optional[str]]
    covariate_name: Mapped[str]
    covariate_value: Mapped[str]
    sample: Mapped["Sample"] = relationship(back_populates="covariates")
