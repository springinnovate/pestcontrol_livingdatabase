import time
from datetime import datetime
from io import StringIO, BytesIO
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

from gee_database_point_sampler import START_DATE, END_DATE, COLLECTION_TEMPORAL_RESOLUTION, NOMINAL_SCALE
from celery_config import make_celery
from database import SessionLocal
from database_model_definitions import LATITUDE, LONGITUDE, YEAR, STUDY_ID
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import jsonify
from flask import make_response
from flask import render_template
from flask import request
from flask import send_file
from flask import url_for
from flask import redirect
from gee_database_point_sampler import initialize_gee
from gee_database_point_sampler import SPATIOTEMPORAL_FN_GRAMMAR
from gee_database_point_sampler import SpatioTemporalFunctionProcessor
from sqlalchemy import distinct, func
from sqlalchemy import select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import and_
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import String
import gee_database_point_sampler
import numpy
import pandas as pd
import redis

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

config = configparser.ConfigParser()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config.update(
    broker_url='redis://redis:6379/0',
    result_backend='redis://redis:6379/0'
)

celery = make_celery(app)
redis_client = redis.Redis(host='redis', port=6379, db=0)


INSTANCE_DIR = '/usr/local/data/'
QUERY_RESULTS_DIR = os.path.join(INSTANCE_DIR, 'results_to_download')
os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)
MAX_SAMPLE_DISPLAY_SIZE = 1000


initialize_gee()


class group_concat_distinct(GenericFunction):
    type = String()
    inherit_cache = True  # Enable caching if the superclass supports it


@compiles(group_concat_distinct, 'sqlite')
def compile_group_concat_distinct(element, compiler, **kw):
    return 'GROUP_CONCAT(DISTINCT %s)' % compiler.process(element.clauses)


@compiles(group_concat_distinct, 'postgresql')
def compile_group_concat_distinct_pg(element, compiler, **kw):
    return 'STRING_AGG(DISTINCT %s, \',\')' % compiler.process(element.clauses)


OPERATION_MAP = {
    '=': lambda field, value: field == value,
    '<': lambda field, value: field < value,
    '>': lambda field, value: field > value
}


def generate_hash_key(data_dict):
    json_data = json.dumps(data_dict, sort_keys=True)
    hash_object = hashlib.sha256(json_data.encode())
    hash_key = hash_object.hexdigest()[:8]
    return hash_key


def nl2br(value):
    return value.replace('\n', '<br>\n')


def to_dict(covariate_list):
    covariate_dict = collections.defaultdict(lambda: None)
    for covariate in covariate_list:
        if covariate is None:
            continue
        if isinstance(covariate, str):
            # Hard-code STUDY_ID
            covariate_dict[STUDY_ID] = covariate
        else:
            covariate_dict[covariate.covariate_defn.name] = covariate.value
    return covariate_dict


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


def _collect_unique_covariate_values(covariates, unique_values_per_covariate):
    for covariate in covariates:
        if isinstance(covariate, str):
            # This will be the STUDY_ID
            unique_values_per_covariate[STUDY_ID].add(covariate)
            continue
        if covariate is None:
            continue
        value = covariate.value
        if value is None or (isinstance(value, float) and numpy.isnan(value)):
            continue
        if isinstance(value, str):
            if value.lower() == 'null':
                continue
            try:
                # Check if value can be converted to float
                float(value)
                unique_values_per_covariate[covariate.covariate_defn.name].add(True)
            except ValueError:
                unique_values_per_covariate[covariate.covariate_defn.name].add(value.lower())
        else:
            # Numeric value; note that it's defined
            unique_values_per_covariate[covariate.covariate_defn.name].add(True)


def calculate_display_tables(session, query, max_sample_size):
    global COVARIATE_STATE
    try:
        covariate_defns = COVARIATE_STATE['covariate_defns']
    except NameError:
        # the celery worker won't call initalize so we can do it here the first time
        initialize_covariates(False)
        covariate_defns = COVARIATE_STATE['covariate_defns']
    sample_covariate_defns = [
        (name, always_display, hidden)
        for name, always_display, hidden, cov_association, show_in_point_table, _ in covariate_defns
        if cov_association == CovariateAssociation.SAMPLE.value or show_in_point_table == 1
    ]

    study_covariate_defns = [
        (name, always_display, hidden)
        for name, always_display, hidden, cov_association, show_in_point_table, _ in covariate_defns
        if cov_association == CovariateAssociation.STUDY.value or show_in_point_table == 1
    ]

    unique_sample_covariate_values = collections.defaultdict(set)
    unique_study_covariate_values = collections.defaultdict(set)

    sample_covariate_list = []
    study_list = []
    study_set = set()
    limited_sample_query = query.limit(max_sample_size)
    for sample, study in limited_sample_query:
        # Collect sample covariates
        sample_covariates = sample.covariates
        sample_covariate_list.append((sample, sample_covariates))
        _collect_unique_covariate_values(sample_covariates, unique_sample_covariate_values)

        # Collect study IDs for later use
        if study.id_key not in study_set:
            study_set.add(study.id_key)
            study_list.append(study)

    study_covariate_list = []
    for study in study_list:
        covariates = study.covariates
        study_covariate_list.append((study, covariates))
        _collect_unique_covariate_values(covariates, unique_study_covariate_values)

    sample_covariate_display_order = [STUDY_ID]
    for name, always_display, hidden in sample_covariate_defns:
        if hidden:
            continue
        if always_display or unique_sample_covariate_values[name]:
            sample_covariate_display_order.append(name)

    # Study covariate display order
    study_covariate_display_order = [STUDY_ID]
    for name, always_display, hidden in study_covariate_defns:
        if hidden:
            continue
        if always_display or unique_study_covariate_values[name]:
            study_covariate_display_order.append(name)

    # Build sample table
    sample_table = []
    for sample, covariate_list in sample_covariate_list:
        covariate_dict = to_dict(covariate_list)
        covariate_dict[STUDY_ID] = sample.study.name  # Include STUDY_ID
        display_row = [covariate_dict.get(name) for name in sample_covariate_display_order]
        sample_table.append(display_row)

    # Build study table
    study_table = []
    for study, covariate_list in study_covariate_list:
        covariate_dict = to_dict(covariate_list)
        covariate_dict[STUDY_ID] = study.name  # Include STUDY_ID
        display_row = [covariate_dict.get(name) for name in study_covariate_display_order]
        study_table.append(display_row)

    return sample_covariate_display_order, sample_table, study_covariate_display_order, study_table


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/initialize_searchable_covariates', methods=['POST'])
def initialize_searchable_covariates():
    data = request.get_json()
    clear_cache = data.get('clear_cache', False)
    initialize_covariates(clear_cache)
    return jsonify(success=True)


def initialize_covariates(clear_cache):
    global COVARIATE_STATE
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    pkcl_filepath = os.path.join(INSTANCE_DIR, 'initialize_searchable_covariates.pkl')
    if os.path.exists(pkcl_filepath):
        if clear_cache:
            os.remove(pkcl_filepath)
        else:
            with open(pkcl_filepath, 'rb') as file:
                COVARIATE_STATE = pickle.load(file)
            LOGGER.info(f'loaded covariate state from {pkcl_filepath}')
            return

    session = SessionLocal()
    COVARIATE_STATE = {}
    LOGGER.debug('starting search for unique values')
    searchable_unique_covariates = (
        session.query(
            CovariateDefn.name,
            CovariateValue.value
        ).filter(
            CovariateDefn.queryable,
            CovariateDefn.search_by_unique == True
        )
        .join(CovariateValue)
    ).yield_per(1000)

    COVARIATE_STATE['searchable_covariates'] = collections.defaultdict(set)
    for index, row in enumerate(searchable_unique_covariates):
        if index % 100000 == 0:
            LOGGER.info(f'on searchable descrete covariates index {index}')
        COVARIATE_STATE['searchable_covariates'][row.name].add(row.value)

    for key, value in COVARIATE_STATE['searchable_covariates'].items():
        COVARIATE_STATE['searchable_covariates'][key] = sorted(value)

    searchable_continuous_covariates = (
        session.query(
            CovariateDefn.name,
            CovariateDefn.covariate_type
        ).filter(
            CovariateDefn.queryable,
            CovariateDefn.search_by_unique == False,
        )
    )
    LOGGER.debug('starting search for continuous covarates')

    for index, row in enumerate(searchable_continuous_covariates):
        if index % 100000 == 0:
            LOGGER.info(f'on searchable continuous covariates index {index}')
        COVARIATE_STATE['searchable_covariates'][row.name] = CovariateType(row.covariate_type).name

    COVARIATE_STATE['n_samples'] = session.query(Sample).count()

    COVARIATE_STATE['country_set'] = [
        x[0] for x in session.query(
            distinct(Geolocation.geolocation_name)).filter(
            Geolocation.geolocation_type == 'COUNTRY').all()]
    COVARIATE_STATE['continent_set'] = [
        x[0] for x in session.query(
            distinct(Geolocation.geolocation_name)).filter(
            Geolocation.geolocation_type == 'CONTINENT').all()]

    # add the study ids manually
    COVARIATE_STATE['searchable_covariates'][STUDY_ID] = [x[0] for x in session.query(distinct(Study.name)).all()]

    # get a list of all the covariates
    covariate_defns = session.query(
        CovariateDefn.name,
        CovariateDefn.always_display,
        CovariateDefn.hidden,
        CovariateDefn.covariate_association,
        CovariateDefn.show_in_point_table,
        CovariateDefn.display_order
    ).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)
    ).all()
    COVARIATE_STATE['covariate_defns'] = covariate_defns

    with open(pkcl_filepath, 'wb') as file:
        pickle.dump(COVARIATE_STATE, file)

    LOGGER.info('sort all the values')

    for key in COVARIATE_STATE['searchable_covariates']:
        COVARIATE_STATE['searchable_covariates'][key] = sorted(COVARIATE_STATE['searchable_covariates'][key])

    LOGGER.info('all done with unique values')


@app.route('/')
def home():
    global COVARIATE_STATE
    return render_template(
        'query_builder.html',
        possible_operations=list(OPERATION_MAP),
        country_set=sorted(COVARIATE_STATE['country_set']),
        continent_set=sorted(COVARIATE_STATE['continent_set']),
        unique_covariate_values=COVARIATE_STATE['searchable_covariates'],
    )


@app.route('/api/n_samples', methods=['POST'])
def n_samples():
    start_time = time.time()
    session = SessionLocal()
    form_as_dict = form_to_dict(request.form)

    query_id = generate_hash_key(form_as_dict)
    redis_key = f"sample_count:{query_id}"
    cached_count = redis_client.get(redis_key)
    if cached_count is not None:
        return cached_count

    filters, filter_text = build_filter(session, form_as_dict)
    sample_query = (
        session.query(Sample.id_key, Study.id_key)
        .join(Sample.study)
        .join(Sample.point)
        .filter(
            *filters
        )
    )
    sample_count = sample_query.count()
    study_query = (
        session.query(Study.id_key)
        .join(Study.samples)
        .join(Sample.point)
        .filter(
            *filters
        ).distinct()
    )
    study_count = study_query.count()

    redis_client.set(query_id, json.dumps(form_as_dict))

    session.close()
    LOGGER.info(f'n_samples query: {filter_text}')

    cached_count = {
        'sample_count': sample_count,
        'study_count': study_count,
        'filter_text': filter_text + f'in {time.time()-start_time:.2f}s'
    }
    redis_client.set(redis_key, json.dumps(cached_count), ex=3600)
    return jsonify(cached_count)


def form_to_dict(form):
    centerPointBuffer = form.get('centerPointBuffer')

    # Get the list of covariates and their counts
    covariates = form.getlist('covariate')
    value_counts = [int(count) for count in form.getlist('valueCounts')]
    values_flat = form.getlist('values')

    # Group values per covariate
    values_grouped = []
    idx = 0
    for count in value_counts:
        group = values_flat[idx:idx+count]
        values_grouped.append(group)
        idx += count

    return {
        'covariate': covariates,
        'values': values_grouped,
        'centerPoint': form.get('centerPoint'),
        'centerPointBuffer': None if not centerPointBuffer else centerPointBuffer,
        'upperLeft': form.get('upperLeft'),
        'lowerRight': form.get('lowerRight'),
        'countrySelect': form.get('countrySelect'),
        'continentSelect': form.get('continentSelect'),
        'minSitesPerStudy': int(form.get('minSitesPerStudy')),
        'sampleSizeMinYears': int(form.get('sampleSizeMinYears')),
        'sampleSizeMinObservationsPerYear': int(form.get('sampleSizeMinObservationsPerYear')),
        'yearRange': form.get('yearRange'),
    }


def build_filter(session, form):
    LOGGER.debug(form)
    fields = form['covariate']
    values_list = form['values']
    filters = []
    filter_text = ''

    # Fetch all covariate_defns at once
    covariate_defns = session.query(
        CovariateDefn.name, CovariateDefn.id_key, CovariateDefn.covariate_association
    ).filter(
        CovariateDefn.name.in_(fields)
    ).all()
    covariate_defn_map = {name: (id_key, association) for name, id_key, association in covariate_defns}

    for field, values in zip(fields, values_list):
        if not field or not values:
            continue
        covariate_defn_id, covariate_association = covariate_defn_map.get(field, (None, None))
        filter_text += f'{field}({covariate_association}) = ' + '|'.join(values) + '\n'
        if field == STUDY_ID:
            filters.append(Study.name.in_(values))
            continue

        covariate_subquery = session.query(CovariateValue)
        covariate_subquery = covariate_subquery.filter(
            CovariateValue.covariate_defn_id == covariate_defn_id,
            CovariateValue.value.in_(values)
        )

        if covariate_association == CovariateAssociation.STUDY.value:
            study_ids_subq = covariate_subquery.with_entities(CovariateValue.study_id).subquery()
            filters.append(Study.id_key.in_(study_ids_subq))
        elif covariate_association == CovariateAssociation.SAMPLE.value:
            sample_ids_subq = covariate_subquery.with_entities(CovariateValue.sample_id).subquery()
            filters.append(Sample.id_key.in_(sample_ids_subq))

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


@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        session = SessionLocal()
        form_as_dict = form_to_dict(request.form)
        filters, filter_text = build_filter(session, form_as_dict)
        LOGGER.info(f'processing query for: {filter_text}')

        query_id = generate_hash_key(form_as_dict)
        redis_client.set(query_id, json.dumps(form_as_dict))

        redis_key = f"sample_count:{query_id}"
        LOGGER.debug(f'sample key: {redis_key}')
        cached_count = redis_client.get(redis_key)
        n_samples = 0
        if cached_count is not None:
            cached_count = json.loads(cached_count)
            if 'sample_count' in cached_count:
                n_samples = cached_count['sample_count']

        session.close()

        return render_template(
            'results_view.html',
            n_samples=n_samples,
            query_id=query_id,
            compiled_query=f'<pre>Filter as text: {filter_text}</pre>',
        )
    except Exception as e:
        LOGGER.exception(f'error with {e}')
        raise


@app.route('/get_data')
def get_data():
    query_id = request.args.get('query_id')
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 100))

    # Retrieve query parameters from Redis
    form_as_dict = json.loads(redis_client.get(query_id))

    # Reconstruct the query
    session = SessionLocal()
    filters, filter_text = build_filter(session, form_as_dict)
    LOGGER.info(f'processing query for: {filter_text} offset {offset}')
    sample_query = (
        session.query(Sample, Study)
        .join(Sample.study)
        .join(Sample.point)
        .filter(
            *filters
        )
        .options(selectinload(Sample.covariates))
    )

    # Process data from offset to offset+limit
    limited_sample_query = sample_query.offset(offset).limit(limit)
    samples_chunk = limited_sample_query.all()
    if not samples_chunk:
        return jsonify({'has_more': False})

    # Initialize data structures
    sample_data_rows = []
    study_data_rows = []
    points = []
    study_set = set()

    # Retrieve columns sent so far from Redis
    sent_columns_key = f"{query_id}_sent_columns"
    sent_columns = redis_client.get(sent_columns_key)
    if sent_columns:
        sent_sample_columns, sent_study_columns = json.loads(sent_columns)
        sent_sample_columns = {name: order for name, order in sent_sample_columns}
        sent_study_columns = {name: order for name, order in sent_study_columns}
    else:
        sent_sample_columns = {}
        sent_study_columns = {}

    study_set_key = f"{query_id}_study_set"
    if offset > 0:
        study_set = set(json.loads(redis_client.get(study_set_key) or "[]"))
    else:
        study_set = set()

    fixed_sample_columns = [
        ('row_number', -4),
        ('study_id', -3),
        ('observation', -2),
        ('latitude', -1),
        ('longitude', 0),
    ]
    for name, order in fixed_sample_columns:
        if name not in sent_sample_columns:
            sent_sample_columns[name] = order

    fixed_study_columns = [
        ('row_number', -1),
        ('study_id', 1),
    ]
    for name, order in fixed_study_columns:
        if name not in sent_study_columns:
            sent_study_columns[name] = order

    for sample, study in samples_chunk:
        # Collect sample covariates
        sample_covariates = sample.covariates
        for cov in sample_covariates:
            if cov is None:
                continue
            cov_name = cov.covariate_defn.name
            if cov_name not in sent_sample_columns:
                display_order = cov.covariate_defn.display_order
                sent_sample_columns[cov_name] = display_order
        # Build sample data row
        sample_row = {
            'row_number': offset + len(sample_data_rows) + 1,
            'observation': sample.observation,
            'latitude': sample.point.latitude,
            'longitude': sample.point.longitude,
            'study_id': sample.study.name
        }
        sample_row.update({
            cov.covariate_defn.name: cov.value
            for cov in sample_covariates if cov is not None})
        sample_data_rows.append(sample_row)

        # Collect point data
        points.append({'lat': sample.point.latitude, 'lng': sample.point.longitude})

        # Collect study data
        if study.id_key not in study_set:
            study_set.add(study.id_key)
            covariates = study.covariates
            LOGGER.debug(covariates)
            for cov in covariates:
                if cov.covariate_defn is None:
                    LOGGER.warn(f'{cov.value} has no defentition')
                    continue
                cov_name = cov.covariate_defn.name
                if cov_name not in sent_study_columns:
                    display_order = cov.covariate_defn.display_order
                    sent_study_columns[cov_name] = display_order
            # Build study data row
            study_row = {
                'row_number': len(study_data_rows) + 1,
                'study_id': study.name
            }
            study_row.update({
                cov.covariate_defn.name: cov.value for cov in covariates
                if cov.covariate_defn is not None})
            study_data_rows.append(study_row)

    # Update the columns sent so far
    sorted_sample_columns = sorted(sent_sample_columns.items(), key=lambda x: x[1])
    sorted_study_columns = sorted(sent_study_columns.items(), key=lambda x: x[1])

    # Store back in Redis
    redis_client.set(sent_columns_key, json.dumps([sorted_sample_columns, sorted_study_columns]))
    redis_client.set(study_set_key, json.dumps(list(study_set)))

    sample_column_names = [name for name, order in sorted_sample_columns]
    study_column_names = [name for name, order in sorted_study_columns]

    # Check if there's more data
    total_samples = sample_query.count()
    has_more = (offset + limit) < total_samples

    # Return the data
    response_data = {
        'sample_columns': sample_column_names,
        'study_columns': study_column_names,
        'sample_data_rows': sample_data_rows,
        'study_data_rows': study_data_rows,
        'points': points,
        'has_more': has_more
    }

    return jsonify(response_data)


@app.route('/admin/covariate', methods=['GET', 'POST'])
def admin_covariate():
    return render_template(
        'admin_covariate.html',
        covariate_association_states=[x.value for x in CovariateAssociation],)


@app.route('/view/covariate', methods=['GET'])
def view_covariate():
    return render_template(
        'covariate_view.html',
        covariate_association_states=[str(x).split('.')[1] for x in CovariateAssociation],)


@app.route('/api/get_covariates', methods=['GET'])
def get_covariates():
    session = SessionLocal()
    covariate_list = session.query(CovariateDefn).order_by(
        CovariateDefn.covariate_association,
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()
    covariates = [{
        'id_key': c.id_key,
        'name': c.name,
        'display_order': c.display_order,
        'description': c.description,
        'always_display': c.always_display,
        'queryable': c.queryable,
        'covariate_association': CovariateAssociation(c.covariate_association).value,
        'hidden': c.hidden,
        'editable_name': c.editable_name,
    } for c in covariate_list]
    session.close()
    return jsonify(success=True, covariates=covariates)


@app.route('/api/update_covariate', methods=['POST'])
def update_covariate():
    session = SessionLocal()
    data = request.json
    for remote_covariate in data['covariates']:
        local_covariate = session.query(CovariateDefn).get(remote_covariate['id_key'])
        local_covariate.name = remote_covariate['name']
        local_covariate.display_order = remote_covariate['display_order']
        local_covariate.description = remote_covariate['description']
        local_covariate.queryable = remote_covariate['queryable']
        local_covariate.always_display = remote_covariate['always_display']
        local_covariate.hidden = remote_covariate['hidden']
    session.commit()

    return get_covariates()


@app.route('/start_download', methods=['POST'])
def start_download():
    query_id = request.form.get('query_id')
    task = _prep_download.apply_async(args=[query_id])
    LOGGER.debug(f'starting prep download task on {task}')
    return jsonify({'task_id': task.id, 'query_id': query_id}), 202


@celery.task
def delete_file(zipfile_path):
    try:
        if os.path.exists(zipfile_path):
            os.remove(zipfile_path)
            LOGGER.info(f"Deleted file: {zipfile_path}")
        else:
            LOGGER.warning(f"File not found for deletion: {zipfile_path}")
    except Exception as e:
        LOGGER.error(f"Error deleting file {zipfile_path}: {e}")


@celery.task(bind=True)
def _prep_download(self, query_id):
    temp_files = []
    try:
        session = SessionLocal()
        LOGGER.info(f'starting prep download with this query id: {query_id}')
        redis_key = f"sample_count:{query_id}"
        cached_count = redis_client.get(redis_key)
        if cached_count is not None:
            total_samples = json.loads(cached_count)['sample_count']
        else:
            total_samples = '-'

        zipfile_path = os.path.join(QUERY_RESULTS_DIR, f'{query_id}.zip')
        query_form_json = redis_client.get(query_id)
        if query_form_json:
            query_form = json.loads(query_form_json)
        else:
            LOGGER.error(f'No query form found for the {query_id}.')
            return 'Error: No query form found.'

        filters, filter_text = build_filter(session, query_form)

        # Collect all covariate names for samples
        sample_covariate_names = list(
            name for (name,) in session.query(CovariateDefn.name)
            .join(CovariateValue, CovariateDefn.id_key == CovariateValue.covariate_defn_id)
            .filter(CovariateValue.sample_id is not None)
            .join(Sample, CovariateValue.sample_id == Sample.id_key)
            .join(Sample.study)
            .filter(*filters)
            .group_by(CovariateDefn.name)
            .order_by(
                CovariateDefn.display_order,
                func.lower(CovariateDefn.name))
            .distinct()
        )

        # Collect all covariate names for studies
        study_covariate_names = list(
            name for (name,) in session.query(CovariateDefn.name)
            .join(CovariateValue, CovariateDefn.id_key == CovariateValue.covariate_defn_id)
            .filter(CovariateValue.study_id is not None)
            .group_by(CovariateDefn.name)
            .order_by(
                CovariateDefn.display_order,
                func.lower(CovariateDefn.name))
            .distinct()
        )

        # Create temporary CSV file paths
        sample_csv_filename = f'sample_data_{query_id}.csv'
        sample_csv_path = os.path.join(QUERY_RESULTS_DIR, sample_csv_filename)
        temp_files.append(sample_csv_path)

        study_csv_filename = f'study_data_{query_id}.csv'
        study_csv_path = os.path.join(QUERY_RESULTS_DIR, study_csv_filename)
        temp_files.append(study_csv_path)

        study_ids = set()

        sample_columns = ['study_id', 'observation', 'latitude', 'longitude', ]
        sample_columns.extend(sample_covariate_names)

        # Prepare the sample query
        batch_size = 1000  # Adjust batch size based on available memory
        sample_query = (
            session.query(Sample, Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(*filters)
            .options(selectinload(Sample.covariates))
        ).yield_per(batch_size)

        batch_rows = []

        first_batch = True
        for index, sample in enumerate(sample_query):
            if index % batch_size == 0:
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': index,
                        'total': total_samples,
                        'query_id': query_id
                    }
                )

            sample_row, study = sample
            study_ids.add(study.id_key)
            row_data = {
                'observation': sample_row.observation,
                'latitude': sample_row.point.latitude,
                'longitude': sample_row.point.longitude,
                'study_id': study.name,
            }
            for cov in sample_row.covariates:
                if cov is None:
                    continue
                cov_name = cov.covariate_defn.name
                row_data[cov_name] = cov.value

            batch_rows.append(row_data)

            # Write the batch when the batch size is reached
            if len(batch_rows) >= batch_size:
                df = pd.DataFrame(batch_rows, columns=sample_columns)
                df.fillna("None", inplace=True)
                df.to_csv(
                    sample_csv_path,
                    mode='a',  # Append mode
                    header=first_batch,  # Write header only for the first batch
                    index=False
                )
                first_batch = False  # Only write header for the first batch
                batch_rows = []  # Clear the batch

        # Write any remaining rows
        if batch_rows:
            df = pd.DataFrame(batch_rows, columns=sample_columns)
            df.fillna("None", inplace=True)
            df.to_csv(
                sample_csv_path,
                mode='a',
                header=first_batch,
                index=False
            )

        # Write study data to CSV file on disk
        study_columns = ['study_id']
        study_columns.extend(study_covariate_names)

        batch_rows = []
        for study in session.query(Study).filter(Study.id_key.in_(study_ids)).options(selectinload(Study.covariates)):
            row_data = {
                'study_id': study.name,
            }
            for cov in study.covariates:
                cov_name = cov.covariate_defn.name
                row_data[cov_name] = cov.value
            row = [row_data.get(col, '') for col in study_columns]
            batch_rows.append(row)
        df = pd.DataFrame(batch_rows, columns=study_columns)
        df.fillna("None", inplace=True)
        df.to_csv(
            study_csv_path,
            mode='a',
            header=True,
            index=False
        )

        # Create ZIP file and add the CSV files
        with zipfile.ZipFile(zipfile_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(sample_csv_path, arcname=sample_csv_filename)
            zf.write(study_csv_path, arcname=study_csv_filename)

        LOGGER.debug(f'{zipfile_path} is created')

        # Schedule deletion of the ZIP file after an hour
        delete_file.apply_async(args=[zipfile_path], countdown=3600)

        return {'query_id': query_id}

    except Exception:
        LOGGER.exception('Error in _prep_download')
        raise  # Reraise the exception to be handled elsewhere

    finally:
        session.close()
        # Delete temporary CSV files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    LOGGER.exception(f'Error deleting temporary file: {temp_file}')


def _wrap_in_quotes_if_needed(value):
    if isinstance(value, str):
        if '"' in value:
            value = value.replace('"', '""')
        if ',' in value or '"' in value:
            return f'"{value}"'
    return value


MAX_EO_POINT_SAMPLES = 5000


def parse_spatiotemporal_fn(spatiotemporal_fn):
    spatiotemporal_fn = spatiotemporal_fn.replace(' ', '')
    grammar_tree = SPATIOTEMPORAL_FN_GRAMMAR.parse(spatiotemporal_fn)
    lexer = SpatioTemporalFunctionProcessor()
    output = lexer.visit(grammar_tree)
    return output


@app.route('/eo_extractor', methods=['GET', 'POST'])
def data_extractor():
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        form_data = request.form.to_dict()
        csv_content = csv_file.read()

        task = gee_data_pull_task.delay(form_data, csv_content, csv_file.filename)
        return redirect(url_for('processing_page', task_id=task.id))

    data_sources = [
        '*GOOGLE/DYNAMICWORLD/V1 crop and landcover table',
        '*MODIS/006/MCD12Q1 landcover table',
        'ECMWF/ERA5/MONTHLY:dewpoint_2m_temperature',
        'ECMWF/ERA5/MONTHLY:maximum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:mean_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:minimum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:total_precipitation',
        'MODIS/061/MCD12Q2:EVI_Amplitude_1',
        'MODIS/061/MCD12Q2:EVI_Area_1',
        'MODIS/061/MCD12Q2:Dormancy_1',
        'MODIS/061/MCD12Q2:Greenup_1',
        'MODIS/061/MCD12Q2:Peak_1',
        'CSP/ERGo/1_0/Global/SRTM_topoDiversity:constant'
    ]

    return render_template(
        'remote_sensed_data_extractor.html',
        latitude_id=LATITUDE,
        longitude_id=LONGITUDE,
        year_id=YEAR,
        max_eo_points=MAX_EO_POINT_SAMPLES,
        data_sources=data_sources,
    )


@app.route('/processing/<task_id>')
def processing_page(task_id):
    """
    Renders a page with a spinner/loader and JS to poll the status endpoint.
    """
    return render_template('processing.html', task_id=task_id)


@app.route('/validate_csv', methods=['POST'])
def validate_csv():
    csv_data = request.form['csv_data']
    invalid_message = ''
    LOGGER.debug(csv_data)
    try:
        # Read first few bytes to validate headers
        first_line = next(iter(csv_data.split('\n')))
        headers = [h.strip() for h in first_line.split(',')]
        valid_headers = all([
            expected_header in headers
            for expected_header in [LATITUDE, LONGITUDE, YEAR]])
        # Check if the first line matches expected headers
        if not valid_headers:
            invalid_message += f'The file does not have the expected headers, got "{headers}"!<br/>'
        # Perform validation and processing
        try:
            csv_file_stream = StringIO(csv_data)
            csv_reader = csv.reader(csv_file_stream)
            _ = next(csv_reader)  # Try to read the header
        except csv.Error as e:
            invalid_message += f'The file is not a valid CSV ({str(e)})!\n'
        return jsonify({
            'valid': False if invalid_message else True,
            'message': invalid_message})
    except Exception as e:
        return jsonify({'valid': False, 'message': str(e)})


@app.route('/download-csv-template')
def download_csv_template():
    # Create a response object and set headers
    response = make_response(','.join([YEAR, LATITUDE, LONGITUDE]))
    response.headers['Content-Disposition'] = 'attachment; filename=eo_sample_template.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response


LOGGER.debug(os.getenv('INIT_COVARIATES'))
if os.getenv('INIT_COVARIATES') == 'True':
    initialize_covariates(False)


@celery.task(bind=True)
def gee_data_pull_task(self, form_data, csv_content, original_filename):
    csv_buffer = BytesIO(csv_content)

    point_features_by_year, point_unique_id_per_year, point_table = (
        gee_database_point_sampler.point_table_to_point_batch(csv_buffer)
    )

    try:
        dataset_id, band_name = form_data.get('data_source').split(':')
    except ValueError:
        dataset_id = form_data.get('data_source')
        band_name = ''

    num_years_avg = int(form_data.get('num_years_avg'))
    seasonality_aggregation_fn = form_data.get('seasonality_aggregation_fn')
    julian_start_day = int(form_data.get('julian_start_day'))
    julian_end_day = int(form_data.get('julian_end_day'))
    spatial_aggregation = form_data.get('spatial_aggregation_fn')
    spatial_radius = int(form_data.get('spatial_radius'))

    sp_tm_agg_op_str = (
        f'spatial_{spatial_aggregation}({spatial_radius};'
        f'years_mean(-{num_years_avg},0;'
        f'julian_{seasonality_aggregation_fn}({julian_start_day},{julian_end_day})))'
    )
    sp_tm_agg_op = parse_spatiotemporal_fn(sp_tm_agg_op_str)

    if not dataset_id.startswith('*'):
        result = gee_database_point_sampler.parse_gee_dataset_info(dataset_id)
        point_id_value_list = gee_database_point_sampler.process_gee_dataset(
            dataset_id,
            band_name,
            result[START_DATE],
            result[END_DATE],
            result[COLLECTION_TEMPORAL_RESOLUTION],
            result[NOMINAL_SCALE],
            point_features_by_year,
            point_unique_id_per_year,
            None,
            sp_tm_agg_op
        )
    else:
        point_id_value_list = gee_database_point_sampler.process_custom_dataset(
            dataset_id,
            point_features_by_year,
            point_unique_id_per_year,
            sp_tm_agg_op
        )

    value_dict = dict(point_id_value_list)
    LOGGER.debug(value_dict)
    for column_name, list_of_tuples in value_dict.items():
        data_dict = dict(list_of_tuples)
        point_table[column_name] = point_table.index.map(data_dict)
        point_table[column_name] = pd.to_numeric(point_table[column_name], errors='ignore')

    csv_output = StringIO()
    point_table.to_csv(csv_output, index=False)
    csv_output.seek(0)
    final_csv = '\ufeff' + csv_output.getvalue()
    return {
        'csv': final_csv,
        'original_filename': original_filename
    }


@app.route('/gee_eo_pull_status/<task_id>')
def gee_eo_pull_status(task_id):
    task = gee_data_pull_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'PENDING'}), 200
    elif task.state == 'STARTED':
        return jsonify({'status': 'PROCESSING'}), 200
    elif task.state == 'SUCCESS':
        LOGGER.debug(f'in gee eo p ull status here is the result: {task.result}')
        return jsonify({
            'status': 'DONE',
            'download_url': url_for('download_result', task_id=task_id, _external=True, _scheme='https')
        }), 200
    elif task.state == 'FAILURE':
        return jsonify({'status': 'ERROR', 'message': str(task.info)}), 200
    else:
        return jsonify({'status': task.state}), 200


@app.route('/download/<task_id>')
def download_result(task_id):
    task = gee_data_pull_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        result = task.result
        csv_str = result['csv']
        original_filename = result['original_filename']

        prefix = os.path.basename(os.path.splitext(original_filename)[0])
        return send_file(
            BytesIO(csv_str.encode('utf-8', 'replace')),
            as_attachment=True,
            mimetype='text/csv',
            download_name=f'{prefix}_remote_sensed_point_table_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'
        )
    return "File not ready or task failed", 400


if __name__ == '__main__':
    app.run(debug=True)
