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


RESPONSE_TYPES = [
    'abundance',
    'activity',
    'production',
]


STUDY_ID = 'study_id'
OBSERVATION = 'observation'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'

REQUIRED_SAMPLE_INPUT_FIELDS = [
    OBSERVATION,
    LATITUDE,
    LONGITUDE,
    STUDY_ID,
    ]

REQUIRED_STUDY_FIELDS = [
    STUDY_ID,
]


class CovariateType(enum.Enum):
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"


class CovariateAssociation(enum.Enum):
    STUDY = "STUDY"
    SAMPLE = "SAMPLE"


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

    def __repr__(self):
        return f'<CovariateDefn(id={self.id_key}, name={self.name})'


covariate_to_sample_association = Table(
    'covariate_to_sample_association', Base.metadata,
    Column(
        'covariate_defn_id',
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
        'covariate_defn_id',
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
    covariate_defn_id: Mapped[int] = mapped_column(
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
    def __repr__(self):
        return f'<CovariateValue(id={self.id_key}, covariate_defn={self.covariate_defn}, value={self.value})'


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[str] = mapped_column(primary_key=True)
    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue",
        secondary=covariate_to_study_association,
        back_populates="studies")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="study")

    def __repr__(self):
        return f'<Study(id={self.id_key}, covariates={self.covariates})'


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
