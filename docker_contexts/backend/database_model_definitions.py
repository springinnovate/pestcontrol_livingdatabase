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
    geolocation_name: Mapped[Optional[str]] = mapped_column()
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
    geolocations: Mapped[Optional[List[GeolocationName]]] = relationship(
        "GeolocationName",
        secondary=geolocation_to_point_association,
        back_populates="points")
    latitude: Mapped[float] = mapped_column(index=True)
    longitude: Mapped[float] = mapped_column(index=True)
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="point")


class ResponseType(Base):
    __tablename__ = 'response_type'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="response_type")


class FunctionalType(Base):
    __tablename__ = 'functional_type'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="functional_type")


class Species(Base):
    __tablename__ = 'species'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="species")


class CropName(Base):
    __tablename__ = 'crop_name'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="crop_name")


class SamplingMethod(Base):
    __tablename__ = 'sampling_method'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="sampling_method")


class CovariateName(Base):
    __tablename__ = 'covariate_name'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, nullable=False)
    covariate_values: Mapped[List["CovariateValue"]] = relationship(
        "CovariateValue", back_populates="covariate_name")


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
        ForeignKey('covariate_name.id_key'), nullable=False)
    value: Mapped[str] = mapped_column(nullable=False)
    covariate_name: Mapped[CovariateName] = relationship(
        "CovariateName", back_populates="covariate_values")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample",
        secondary=covariate_to_study_association,
        back_populates="covariates")


study_doi_association = Table(
    'study_doi_association', Base.metadata,
    Column('study_id', Integer, ForeignKey('study.id_key'), primary_key=True),
    Column('doi_id', Integer, ForeignKey('doi.id_key'), primary_key=True)
)


class DOI(Base):
    __tablename__ = 'doi'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[str] = mapped_column(index=True)
    studies: Mapped[List["Study"]] = relationship(
        "Study", secondary=study_doi_association, back_populates="paper_dois")


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[str] = mapped_column(index=True)
    study_metadata: Mapped[Optional[str]] = mapped_column(index=True)
    paper_dois: Mapped[List[DOI]] = relationship(
        "DOI", secondary=study_doi_association, back_populates="studies")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="study")


class EarthObservationSource(Base):
    __tablename__ = 'earth_observation_source'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    metadata: Mapped[str] = mapped_column(index=True)
    values: Mapped[List["EarthObservationValue"]] = relationship(
        "EarthObservationValue", back_populates="earth_observation_source")


class EarthObservationValue(Base):
    __tablename__ = 'earth_observation_value'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[float] = mapped_column()
    earth_observation_source: Mapped[EarthObservationSource] = relationship(
        "EarthObservationSource", back_populates="values")
    sample: Mapped["Sample"] = relationship(
        "Sample", back_populates="earth_observation_values")


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)

    response_id: Mapped[int] = mapped_column(ForeignKey('response_type.id_key'), index=True)
    response_type: Mapped[ResponseType] = relationship(ResponseType, back_populates="samples")

    species_id: Mapped[int] = mapped_column(ForeignKey('species.id_key'), index=True)
    species: Mapped[Species] = relationship(Species, back_populates="samples")

    point_id: Mapped[int] = mapped_column(ForeignKey('point.id_key'), index=True)
    point: Mapped[Point] = relationship(Point, back_populates="samples")

    functional_type_id: Mapped[int] = mapped_column(ForeignKey('functional_type.id_key'), index=True)
    functional_type: Mapped[FunctionalType] = relationship(FunctionalType, back_populates="samples")

    crop_name_id: Mapped[int] = mapped_column(ForeignKey('crop_name.id_key'), index=True)
    crop_name: Mapped[CropName] = relationship(CropName, back_populates="samples")

    sampling_method_id: Mapped[int] = mapped_column(ForeignKey('sampling_method.id_key'), index=True)
    sampling_method: Mapped[SamplingMethod] = relationship(SamplingMethod, back_populates="samples")

    study_id: Mapped[int] = mapped_column(ForeignKey('study.id_key'), index=True)
    study: Mapped[Study] = relationship("Study", back_populates="samples")

    sampling_effort: Mapped[int] = mapped_column(index=True)

    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue",
        secondary=covariate_to_study_association,
        back_populates="samples")

    observation: Mapped[float] = mapped_column(index=True)
    year: Mapped[int] = mapped_column(index=True)
    earth_observation_values: Mapped[List[EarthObservationValue]] = \
        relationship("EarthObservationValue", back_populates="sample")
