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
from sqlalchemy import Column, Integer, UniqueConstraint, Index, Table


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

STUDY_ID = 'study_id'

STUDY_LEVEL_VARIABLES = [
    STUDY_ID,
    'Data contributor',
    'Data contributor contact info',
    'Paper(s) DOI',
    'Metadata',
    ('Response types', RESPONSE_TYPES),
    (COORDINATE_PRECISION_FIELD, COORDINATE_PRECISION),
]

MANAGER = 'manager'
YEAR = 'year'
MONTH = 'month'
DAY = 'day'
TIME = 'time'
REPLICATE = 'replicate'
SAMPLING_EFFORT = 'sampling_effort'
OBSERVATION = 'observation'
OBSERVER_ID = 'observer_id'
RESPONSE_VARIABLE = 'response_variable'
UNITS = 'units'
SAMPLING_METHOD = 'sampling_method'
SAMPLER_TYPE = 'sampler_type'
FUNCTIONAL_TYPE = 'functional_type'
CROP_COMMERCIAL_NAME = 'crop_commercial_name'
CROP_LATIN_NAME = 'crop_latin_name'
GROWTH_STAGE_OF_CROP_AT_SAMPLING = 'growth_stage_of_crop_at_sampling'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'


BASE_FIELDS = [
    LATITUDE,
    LONGITUDE,
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
    Column(STUDY_ID, Integer, ForeignKey('study.id_key'), primary_key=True),
    Column('doi_id', Integer, ForeignKey('doi.id_key'), primary_key=True)
)


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[str]
    study_metadata: Mapped[Optional[str]]
    paper_dois = relationship(
        "DOI", secondary=StudyDOIAssociation, back_populates="studies")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="study")


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
    point_id: Mapped[int] = mapped_column(ForeignKey('point.id_key'))
    study: Mapped[Study] = relationship("Study", back_populates="samples")
    point: Mapped["Point"] = relationship("Point", back_populates="samples")
    manager: Mapped[Optional[str]]
    year: Mapped[Optional[int]]
    month: Mapped[Optional[int]]
    day: Mapped[Optional[int]]
    time: Mapped[Optional[str]]
    replicate: Mapped[Optional[str]]
    sampling_effort: Mapped[Optional[str]]
    observation: Mapped[Optional[str]]
    observer_id: Mapped[Optional[str]]
    response_type: Mapped[Optional[str]]
    response_variable: Mapped[Optional[str]]
    units: Mapped[Optional[str]]
    sampling_method: Mapped[Optional[str]]
    sampler_type: Mapped[Optional[str]]
    functional_type: Mapped[Optional[str]]
    crop_commercial_name: Mapped[Optional[str]]
    crop_latin_name: Mapped[Optional[str]]
    abundance_class: Mapped[Optional[str]]
    order: Mapped[Optional[str]]
    family: Mapped[Optional[str]]
    genus: Mapped[Optional[str]]
    species: Mapped[Optional[str]]
    subspecies: Mapped[Optional[str]]
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


class Point(Base):
    __tablename__ = 'point'
    __table_args__ = (
        UniqueConstraint('latitude', 'longitude', name='uix_lat_long'),
        Index('idx_lat_long', 'latitude', 'longitude'),
    )
    id_key: Mapped[int] = mapped_column(primary_key=True)
    samples: Mapped[List["Sample"]] = relationship(
        back_populates='point')
    latitude: Mapped[float]
    longitude: Mapped[float]
    country: Mapped[Optional[str]]
    continent: Mapped[Optional[str]]


class Covariate(Base):
    __tablename__ = 'covariate'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey('sample.id_key'))
    covariate_name: Mapped[Optional[str]]
    covariate_value: Mapped[Optional[str]]
    covariate_category: Mapped[Optional[str]]
    sample: Mapped["Sample"] = relationship(back_populates="covariates")
