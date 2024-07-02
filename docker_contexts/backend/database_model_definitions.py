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
    study_id: Mapped[int] = mapped_column(ForeignKey('study.id_key'), index=True)
    point_id: Mapped[int] = mapped_column(ForeignKey('point.id_key'), index=True)
    study: Mapped[Study] = relationship("Study", back_populates="samples")
    point: Mapped["Point"] = relationship("Point", back_populates="samples")
    manager: Mapped[Optional[str]] = mapped_column(index=True)
    year: Mapped[Optional[int]] = mapped_column(index=True)
    month: Mapped[Optional[int]] = mapped_column(index=True)
    day: Mapped[Optional[int]] = mapped_column(index=True)
    time: Mapped[Optional[str]] = mapped_column(index=True)
    replicate: Mapped[Optional[str]] = mapped_column(index=True)
    sampling_effort: Mapped[Optional[str]] = mapped_column(index=True)
    observation: Mapped[Optional[str]] = mapped_column(index=True)
    observer_id: Mapped[Optional[str]] = mapped_column(index=True)
    response_type: Mapped[Optional[str]] = mapped_column(index=True)
    response_variable: Mapped[Optional[str]] = mapped_column(index=True)
    units: Mapped[Optional[str]] = mapped_column(index=True)
    sampling_method: Mapped[Optional[str]] = mapped_column(index=True)
    sampler_type: Mapped[Optional[str]] = mapped_column(index=True)
    functional_type: Mapped[Optional[str]] = mapped_column(index=True)
    crop_commercial_name: Mapped[Optional[str]] = mapped_column(index=True)
    crop_latin_name: Mapped[Optional[str]] = mapped_column(index=True)
    abundance_class: Mapped[Optional[str]] = mapped_column(index=True)
    order: Mapped[Optional[str]] = mapped_column(index=True)
    family: Mapped[Optional[str]] = mapped_column(index=True)
    genus: Mapped[Optional[str]] = mapped_column(index=True)
    species: Mapped[Optional[str]] = mapped_column(index=True)
    subspecies: Mapped[Optional[str]] = mapped_column(index=True)
    morphospecies: Mapped[Optional[str]] = mapped_column(index=True)
    life_stage: Mapped[Optional[str]] = mapped_column(index=True)
    pest_class: Mapped[Optional[str]] = mapped_column(index=True)
    pest_order: Mapped[Optional[str]] = mapped_column(index=True)
    pest_family: Mapped[Optional[str]] = mapped_column(index=True)
    pest_species: Mapped[Optional[str]] = mapped_column(index=True)
    pest_sub_species: Mapped[Optional[str]] = mapped_column(index=True)
    pest_morphospecies: Mapped[Optional[str]] = mapped_column(index=True)
    pest_life_stage: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_class: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_order: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_family: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_species: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_sub_species: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_morphospecies: Mapped[Optional[str]] = mapped_column(index=True)
    enemy_lifestage: Mapped[Optional[str]] = mapped_column(index=True)
    growth_stage_of_crop_at_sampling: Mapped[Optional[str]] = mapped_column(index=True)
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
    sample_id: Mapped[int] = mapped_column(ForeignKey('sample.id_key'), index=True)
    covariate_name: Mapped[Optional[str]]
    covariate_value: Mapped[Optional[str]]
    covariate_category: Mapped[Optional[str]]
    sample: Mapped["Sample"] = relationship(back_populates="covariates")


SAMPLE_DISPLAY_FIELDS = [
    Sample.response_type,
    Sample.response_variable,
    Sample.sampling_method,
    Sample.observation,
    Sample.units,
    Sample.functional_type,
    Point.latitude,
    Point.longitude,
    Point.country,
    Sample.year,
    Sample.study,
    Sample.point,
    Sample.manager,
    Sample.month,
    Sample.day,
    Sample.time,
    Sample.replicate,
    Sample.sampling_effort,
    Sample.observer_id,
    Sample.sampler_type,
    Sample.crop_commercial_name,
    Sample.growth_stage_of_crop_at_sampling,
]

FIELDS_BY_RESPONSE_TYPE = {
    'abundance': [
        Sample.crop_latin_name,
        Sample.abundance_class,
        Sample.order,
        Sample.family,
        Sample.genus,
        Sample.species,
        Sample.subspecies,
        Sample.morphospecies,
        Sample.life_stage,
        ],
    'activity': [
        Sample.pest_class,
        Sample.pest_order,
        Sample.pest_family,
        Sample.pest_species,
        Sample.pest_sub_species,
        Sample.pest_morphospecies,
        Sample.pest_life_stage,
        Sample.enemy_class,
        Sample.enemy_order,
        Sample.enemy_family,
        Sample.enemy_species,
        Sample.enemy_sub_species,
        Sample.enemy_morphospecies,
        Sample.enemy_lifestage,
        ],
    'production': [],
}


if len(RESPONSE_TYPES) != len(set(RESPONSE_TYPES).union(set(FIELDS_BY_RESPONSE_TYPE))):
    raise ValueError(
        'Response types and fields by response types do not have the same '
        'fields in database_model_definitions.py, open that file and find out '
        'what is going on.')
