import time
from io import StringIO
import collections
import configparser
import csv
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import zipfile

import numpy
from database import SessionLocal
from database_model_definitions import REQUIRED_STUDY_FIELDS, REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import OBSERVATION, LATITUDE, LONGITUDE
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import send_file
from sqlalchemy import select, text
from sqlalchemy import distinct, func
from sqlalchemy.engine import Row
from sqlalchemy.sql import and_, or_, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import String
from sqlalchemy.orm import subqueryload
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import joinedload, contains_eager, selectinload

STUDY_ID = 1

OPERATION_MAP = {
    '=': lambda field, value: field == value,
    '<': lambda field, value: field < value,
    '>': lambda field, value: field > value
}


def build_filter(session, form):
    fields = form['covariate']
    operations = form['operation']
    values = form['value']
    filters = []
    filter_text = ''

    # Fetch all covariate_defns at once
    covariate_defns = session.query(
        CovariateDefn.name, CovariateDefn.covariate_association
    ).filter(
        CovariateDefn.name.in_(fields)
    ).all()
    covariate_defn_map = dict(covariate_defns)

    for field, operation, value in zip(fields, operations, values):
        if not field or not value:
            continue
        covariate_association = covariate_defn_map.get(field)
        filter_text += f'{field}({covariate_association}) {operation} {value}\n'
        if field == STUDY_ID:
            filters.append(Study.name == value)
            continue

        covariate_subquery = session.query(CovariateValue)
        covariate_subquery = covariate_subquery.join(CovariateDefn)
        covariate_subquery = covariate_subquery.filter(
            CovariateDefn.name == field,
            OPERATION_MAP[operation](CovariateValue.value, value)
        )

        if covariate_association == CovariateAssociation.STUDY.value:
            study_ids = select(covariate_subquery.with_entities(
                CovariateValue.study_id).subquery())
            filters.append(Study.id_key.in_(study_ids))
        elif covariate_association == CovariateAssociation.SAMPLE.value:
            sample_ids = select(covariate_subquery.with_entities(
                CovariateValue.sample_id).subquery())
            filters.append(Sample.id_key.in_(sample_ids))

    ul_lat = None
    center_point = form['centerPoint']
    if center_point is not None:
        center_point = center_point.strip()
        if center_point != '':
            m = re.match(r"[(]?([^, \t]+)[, \t]+([^, )]+)[\)]?", center_point)
            lat, lng = [float(v) for v in m.group(1, 2)]
            center_point_buffer = float(
                form['centerPointBuffer'].strip())/2
            ul_lat = lat+center_point_buffer/2
            lr_lat = lat-center_point_buffer/2
            ul_lng = lng-center_point_buffer/2
            lr_lng = lng+center_point_buffer/2

            filter_text += f'center point at ({lat},{lng}) + +/-{center_point_buffer}\n'

    upper_left_point = form['upperLeft']
    if upper_left_point is not None:
        upper_left_point = form['upperLeft'].strip()
        lower_right_point = form['lowerRight'].strip()
        if upper_left_point != '':
            m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", upper_left_point)
            ul_lat, ul_lng = [float(v) for v in m.group(1, 2)]
            m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", lower_right_point)
            lr_lat, lr_lng = [float(v) for v in m.group(1, 2)]

            filter_text += f'bounding box ({ul_lat},{ul_lng}) ({lr_lat},{lr_lng})\n'

    country_select = form['countrySelect']
    if country_select:
        geolocation_subquery = (
            session.query(Point.id_key)
            .join(Point.geolocations)
            .filter(Geolocation.geolocation_name == country_select)
        ).subquery()
        filters.append(Point.id_key.in_(select(geolocation_subquery)))
        filter_text += f'country is {country_select}\n'

    continent_select = form['continentSelect']
    if continent_select:
        geolocation_subquery = (
            session.query(Point.id_key)
            .join(Point.geolocations)
            .filter(Geolocation.geolocation_name == continent_select)
        ).subquery()
        filters.append(Point.id_key.in_(select(geolocation_subquery)))
        filter_text += f'continent is {continent_select}\n'

    if ul_lat is not None:
        filters.append(and_(
            Point.latitude <= ul_lat,
            Point.latitude >= lr_lat,
            Point.longitude >= ul_lng,
            Point.longitude <= lr_lng))

    min_sites_per_study = int(form['minSitesPerStudy'])
    if min_sites_per_study:
        min_sites_per_study = int(min_sites_per_study)
        filter_text += f'min sites per study {min_sites_per_study}\n'
        min_sites_per_study_subquery = (
            session.query(
                Study.id_key,
                func.count(func.distinct(Sample.point_id))
            )
            .join(Study.samples)
            .group_by(Study.id_key)
            .having(func.count(func.distinct(Sample.point_id)) >= min_sites_per_study)
            .subquery()
        )
        filters.append(
            Study.id_key == min_sites_per_study_subquery.c.id_key)

    sample_size_min_years = int(form['sampleSizeMinYears'])
    if sample_size_min_years > 0:
        filter_text += f'min sample size min years {sample_size_min_years}\n'
        valid_study_ids_subquery = (
            session.query(Sample.study_id)
            .group_by(Sample.study_id)
            .having(func.count(func.distinct(Sample.year)) >= sample_size_min_years)
            .subquery()
        )
        filters.append(Sample.study_id.in_(select(valid_study_ids_subquery)))

    min_observations_per_year = int(form['sampleSizeMinObservationsPerYear'])
    if min_observations_per_year > 0:
        filter_text += f'min observations per year {min_observations_per_year}\n'

        counts_per_study_year = (
            session.query(
                Sample.study_id.label('study_id'),
                Sample.year.label('year'),
                func.count(Sample.id_key).label('samples_per_year')
            )
            .group_by(Sample.study_id, Sample.year)
            .subquery()
        )

        valid_study_ids_subquery = (
            session.query(counts_per_study_year.c.study_id)
            .group_by(counts_per_study_year.c.study_id)
            .having(
                func.min(counts_per_study_year.c.samples_per_year) >= min_observations_per_year
            )
            .subquery()
        )
        filters.append(Sample.study_id.in_(valid_study_ids_subquery))

    year_range = form['yearRange']
    if year_range:
        filter_text += 'years in {' + year_range + '}\n'
        year_set = extract_years(year_range)
        year_subquery_sample_ids = (
            session.query(Sample.id_key)
            .filter(Sample.year.in_(year_set)).subquery())
        filters.append(
            Sample.id_key.in_(year_subquery_sample_ids))
    return filters, filter_text

def extract_years(year_string):
    """Process a comma/space separated string of years into year set."""
    years = set()
    parts = re.split(r'[,\s]+', year_string)

    for part in parts:
        part = part.strip()  # Remove leading and trailing whitespace

        if '-' in part:  # Check if the part is a range
            start_year, end_year = map(int, part.split('-'))
            years.update(range(start_year, end_year + 1))  # Add all years in the range
        else:
            years.add(int(part))  # Add individual year

    return sorted(years)


def main():
    session = SessionLocal()

    filters, filter_text = build_filter(session, {
        'covariate': ['Response_variable'],
        'operation': ['='],
        'value': ['abundance'],
        'centerPoint': None,
        'upperLeft': None,
        'upperLeft': None,
        'lowerRight': None,
        'countrySelect': None,
        'continentSelect': None,
        'minSitesPerStudy': 0,
        'sampleSizeMinYears': 0,
        'sampleSizeMinObservationsPerYear': 0,
        'yearRange': None,
        })
    print(f'these are the filters: {filters}')
    print(filter_text)
    start_time = time.time()
    sample_query = (
        session.query(Sample.id_key, Study.id_key)
        .join(Sample.study)
        .join(Sample.point)
        .filter(
            *filters
        )
    )
    sample_count = sample_query.count()
    print(f'sample {time.time()-start_time:.2f}s')
    study_query = (
        session.query(Study.id_key)
        .join(Study.samples)
        .join(Sample.point)
        .filter(
            *filters
        ).distinct()
    )
    study_count = study_query.count()
    print(f'study {time.time()-start_time:.2f}s')
    session.close()

    print(sample_count)
    print(study_count)


if __name__ == '__main__':
    main()
