"""Database definitions for news articles and their classifications.

Defined from:
https://docs.google.com/spreadsheets/d/1yZwc7fPB0kHI9F5jdgUKuNflgkQF7SHS/edit#gid=1487741928
"""
import enum

from sqlalchemy import Column, Integer, UniqueConstraint, Index, Table, String, Boolean, JSON, Enum
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List
from typing import Optional


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


geolocation_to_point_association = Table(
    'geolocation_to_point_association', Base.metadata,
    Column(
        'point_id',
        ForeignKey('point.id_key'),
        primary_key=True),
    Column(
        'geolocation_id',
        ForeignKey('geolocation_name.id_key'),
        primary_key=True)
)


class GeolocationName(Base):
    __tablename__ = 'geolocation_name'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    geolocation_name: Mapped[str] = mapped_column(unique=True, index=True)
    points: Mapped[list["Point"]] = relationship(
        "Point",
        secondary=geolocation_to_point_association,
        back_populates="geolocations")


class Point(Base):
    __tablename__ = 'point'
    __table_args__ = (
        UniqueConstraint('latitude', 'longitude', name='uix_lat_long'),
        Index('idx_lat_long', 'latitude', 'longitude')
    )
    id_key: Mapped[int] = mapped_column(primary_key=True)
    latitude: Mapped[float] = mapped_column(index=True)
    longitude: Mapped[float] = mapped_column(index=True)
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="point")
    geolocations: Mapped[Optional[List[GeolocationName]]] = relationship(
        "GeolocationName",
        secondary=geolocation_to_point_association,
        back_populates="points")


class CovariateType(enum.Enum):
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"


class CovariateAssociation(enum.Enum):
    STUDY = "STUDY"
    SAMPLE = "SAMPLE"


class CovariateDefn(Base):
    __tablename__ = 'covariate_defn'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    display_order: Mapped[float] = mapped_column(default=0)
    description: Mapped[Optional[str]] = mapped_column(
        String, default='no description provided')
    queryable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    always_display: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    condition: Mapped[dict] = mapped_column(JSON, default=None)
    covariate_type: Mapped[CovariateType] = mapped_column(
        Enum(CovariateType), nullable=False, index=True)
    covariate_association: Mapped[CovariateAssociation] = mapped_column(
        Enum(CovariateAssociation), nullable=False, index=True)
    covariate_values: Mapped[List["CovariateValue"]] = relationship(
        "CovariateValue", back_populates="covariate_defn")


covariate_to_sample_association = Table(
    'covariate_to_sample_association', Base.metadata,
    Column(
        'covariate_id',
        ForeignKey('covariate_value.id_key'),
        primary_key=True),
    Column(
        'sample_id',
        ForeignKey('sample.id_key'),
        primary_key=True)
)

covariate_to_study_association = Table(
    'covariate_to_study_association', Base.metadata,
    Column(
        'covariate_id',
        ForeignKey('covariate_value.id_key'),
        primary_key=True),
    Column(
        'study_id',
        ForeignKey('study.id_key'),
        primary_key=True)
)


class CovariateValue(Base):
    __tablename__ = 'covariate_value'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    covariate_id: Mapped[int] = mapped_column(
        ForeignKey('covariate_defn.id_key'))
    value: Mapped[str] = mapped_column(unique=True, index=True)
    covariate_defn: Mapped[CovariateDefn] = relationship(
        "CovariateDefn", back_populates="covariate_values")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample",
        secondary=covariate_to_sample_association,
        back_populates="covariates")
    studies: Mapped[List["Study"]] = relationship(
        "Study",
        secondary=covariate_to_study_association,
        back_populates="covariates")


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[str] = mapped_column(primary_key=True)
    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue",
        secondary=covariate_to_study_association,
        back_populates="studies")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="study")


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)

    point_id: Mapped[int] = mapped_column(
        ForeignKey('point.id_key'), index=True)
    point: Mapped[Point] = relationship(
        Point, back_populates="samples")

    study_id: Mapped[str] = mapped_column(
        ForeignKey('study.id_key'), index=True)
    study: Mapped[Study] = relationship(
        "Study", back_populates="samples")

    observation: Mapped[float] = mapped_column(index=True)

    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue",
        secondary=covariate_to_sample_association,
        back_populates="samples")


REQUIRED_SAMPLE_INPUT_FIELDS = [
    'latitude',
    'longitude',
    'observation',
    'study_id',
    ]

REQUIRED_STUDY_FIELDS = [
    'study_id',
]
