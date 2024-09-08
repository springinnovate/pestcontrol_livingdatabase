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
YEAR = 'year'

REQUIRED_SAMPLE_INPUT_FIELDS = [
    OBSERVATION,
    LATITUDE,
    LONGITUDE,
    STUDY_ID,
    YEAR,
]

REQUIRED_STUDY_FIELDS = [
    STUDY_ID,
]


class CovariateType(enum.Enum):
    STRING = 0
    FLOAT = 1
    INTEGER = 2


class CovariateAssociation(enum.Enum):
    STUDY = 0
    SAMPLE = 1


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


class Geolocation(Base):
    __tablename__ = 'geolocation_name'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    geolocation_name: Mapped[str] = mapped_column(nullable=False, index=True)
    geolocation_type: Mapped[str] = mapped_column(nullable=False, index=True)
    points: Mapped[list["Point"]] = relationship(
        "Point",
        secondary=geolocation_to_point_association,
        back_populates="geolocations")

    def __repr__(self):
        return (
            f'<Geolocation(id={self.id_key}, '
            f'geolocation_name={self.geolocation_name}>')


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
    geolocations: Mapped[List[Geolocation]] = relationship(
        "Geolocation",
        secondary=geolocation_to_point_association,
        back_populates="points")

    def __repr__(self):
        return (
            f'<Point(id={self.id_key}, latitude={self.latitude}, '
            f'longitude={self.longitude}, geolocations={self.geolocations}>')


class CovariateDefn(Base):
    __tablename__ = 'covariate_defn'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    editable_name: Mapped[bool] = mapped_column(index=True, nullable=False)
    display_order: Mapped[float] = mapped_column(default=0)
    description: Mapped[Optional[str]] = mapped_column(
        String, default='no description provided')
    queryable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    always_display: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    hidden: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    show_in_point_table: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    search_by_unique: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True)
    covariate_type: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True)
    covariate_association: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True)
    covariate_values: Mapped[List["CovariateValue"]] = relationship(
        "CovariateValue", back_populates="covariate_defn")

    def __repr__(self):
        return f'<CovariateDefn(id={self.id_key}, name={self.name})'


class CovariateValue(Base):
    __tablename__ = 'covariate_value'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False, index=True)

    covariate_defn_id: Mapped[int] = mapped_column(
        ForeignKey('covariate_defn.id_key'))
    covariate_defn: Mapped[int] = relationship(
        "CovariateDefn", back_populates="covariate_values")

    study_id: Mapped[Optional[int]] = mapped_column(ForeignKey('study.id_key'))
    study: Mapped[Optional['Study']] = relationship("Study", back_populates="covariates")

    sample_id: Mapped[Optional[int]] = mapped_column(ForeignKey('sample.id_key'), index=True)
    sample: Mapped[Optional['Sample']] = relationship("Sample", back_populates="covariates")

    def __repr__(self):
        return f'<CovariateValue(id={self.id_key}, covariate_defn={self.covariate_defn}, value={self.value})>'


class Study(Base):
    __tablename__ = 'study'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue", back_populates="study")
    samples: Mapped[List["Sample"]] = relationship(
        "Sample", back_populates="study")

    def __repr__(self):
        return f'<Study(id={self.id_key}/{self.name}, covariates={self.covariates})>'


class Sample(Base):
    __tablename__ = 'sample'
    id_key: Mapped[int] = mapped_column(primary_key=True)

    observation: Mapped[float] = mapped_column(index=True)

    covariates: Mapped[List[CovariateValue]] = relationship(
        "CovariateValue", back_populates="sample")

    point_id: Mapped[int] = mapped_column(
        ForeignKey('point.id_key'), index=True)
    point: Mapped[Point] = relationship(
        Point, back_populates="samples")

    study_id: Mapped[str] = mapped_column(
        ForeignKey('study.id_key'), index=True)
    study: Mapped[Study] = relationship(
        "Study", back_populates="samples")

    def __repr__(self):
        return (
            f'<Sample(id={self.id_key}, point={self.point}, '
            f'study_id={self.study_id}, observation={self.observation} '
            f'covariates={self.covariates}>')
