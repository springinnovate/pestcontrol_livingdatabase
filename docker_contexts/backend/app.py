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

from celery_config import make_celery
from database import SessionLocal
from database_model_definitions import OBSERVATION, LATITUDE, LONGITUDE, YEAR, STUDY_ID
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import jsonify
from flask import make_response
from flask import render_template
from flask import request
from flask import send_file
from flask import url_for
from gee_database_point_sampler import initialize_gee
from gee_database_point_sampler import SPATIOTEMPORAL_FN_GRAMMAR
from gee_database_point_sampler import SpatioTemporalFunctionProcessor
from gee_database_point_sampler import UNIQUE_ID
from sqlalchemy import distinct, func
from sqlalchemy import select, text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import and_
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import String
import ee
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
    hash_key = hash_object.hexdigest()
    return hash_key


def nl2br(value):
    return value.replace('\n', '<br>\n')


def to_dict(covariate_list):
    covariate_dict = collections.defaultdict(lambda: None)
    for covariate in covariate_list:
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
        if isinstance(COVARIATE_STATE['searchable_covariates'][row.name], list):
            print(f'{row.name} is a list???')
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

@app.route('/test')
def test():
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
    return (sample_count, study_count)


@app.route('/api/n_samples', methods=['POST'])
def n_samples():
    start_time = time.time()
    session = SessionLocal()
    form_as_dict = form_to_dict(request.form)
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
    session.close()
    print(filter_text)
    return jsonify({
        'sample_count': sample_count,
        'study_count': study_count,
        'filter_text': filter_text + f'in {time.time()-start_time:.2f}s'
    })


def form_to_dict(form):
    centerPointBuffer = form.get('centerPointBuffer')

    return {
        'covariate': form.getlist('covariate'),
        'operation': form.getlist('operation'),
        'value': form.getlist('value'),
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


@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        session = SessionLocal()
        form_as_dict = form_to_dict(request.form)
        filters, filter_text = build_filter(session, form_as_dict)
        sample_query = (
            session.query(Sample, Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .options(selectinload(Sample.covariates))
        )

        (sample_covariate_display_order, sample_table,
         study_covariate_display_order, study_table) = calculate_display_tables(
            session, sample_query, max_sample_size=MAX_SAMPLE_DISPLAY_SIZE)

        # add the lat/lng points and observation to the sample
        sample_covariate_display_order = [
            OBSERVATION, LATITUDE, LONGITUDE] + sample_covariate_display_order
        for (sample_row, _), display_row in zip(sample_query.limit(MAX_SAMPLE_DISPLAY_SIZE), sample_table):
            display_row[:] = [
                sample_row.observation,
                sample_row.point.latitude,
                sample_row.point.longitude] + display_row

        unique_points = {sample[0].point for sample in sample_query.limit(MAX_SAMPLE_DISPLAY_SIZE)}
        points = [
            {"lat": p.latitude, "lng": p.longitude}
            for p in unique_points]

        session.close()
        query_id = generate_hash_key(form_as_dict)
        redis_client.set(query_id, json.dumps(form_as_dict))

        return render_template(
            'results_view.html',
            study_headers=study_covariate_display_order,
            study_table=study_table,
            sample_headers=sample_covariate_display_order,
            sample_table=sample_table,
            points=points,
            compiled_query=f'<pre>Filter as text: {filter_text}</pre>',
            max_samples=MAX_SAMPLE_DISPLAY_SIZE,
            expected_samples=sample_query.count(),
            query_id=query_id,
        )
    except Exception as e:
        LOGGER.exception(f'error with {e}')
        raise


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


@app.route('/prep_download')
def prep_download():
    query_id = request.args.get('query_id', default=None, type=str)
    task = _prep_download.delay(query_id)
    return jsonify({'task_id': task.id}), 202


@app.route('/check_task/<task_id>')
def check_task(task_id):
    task = _prep_download.AsyncResult(task_id)
    LOGGER.debug(f'status for {task_id} is {task.state}')
    if task.state == 'PENDING':
        return jsonify({'status': 'Task is pending'}), 202
    elif task.state == 'FAILURE':
        return jsonify({'status': 'Task failed', 'error': str(task.info)}), 500
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'Task completed', 'result_url': url_for('download_file', task_id=task_id)}), 200
    elif task.state == 'STARTED':
        return jsonify({'status': 'Task is started'}), 202


@app.route('/download_file/<task_id>')
def download_file(task_id):
    # Implement logic to send the file to the client
    file_path = os.path.join(QUERY_RESULTS_DIR, f'{task_id}.zip')
    return send_file(file_path, as_attachment=True)


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


@celery.task
def _prep_download(task_id):
    try:
        session = SessionLocal()
        LOGGER.info(f'starting prep download with this query id: {task_id}')
        zipfile_path = os.path.join(QUERY_RESULTS_DIR, f'{task_id}.zip')
        if os.path.exists(zipfile_path):
            return f"File {zipfile_path} already existed."
        query_form_json = redis_client.get(task_id)
        if query_form_json:
            query_form = json.loads(query_form_json)
        else:
            LOGGER.error(f'No query form found for the {task_id}.')
            return 'Error: No query form found.'
        filters, filter_text = build_filter(session, query_form)

        sample_query = (
            session.query(Sample, Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .options(selectinload(Sample.covariates))
        ).yield_per(1000)

        (sample_covariate_display_order, sample_table,
         study_covariate_display_order, study_table) = calculate_display_tables(
            session, sample_query, max_sample_size=50000)

        sample_covariate_display_order = [
            OBSERVATION, LATITUDE, LONGITUDE] + sample_covariate_display_order

        with zipfile.ZipFile(zipfile_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            sample_table_io = StringIO()

            writer = csv.writer(sample_table_io)
            writer.writerow(sample_covariate_display_order)

            for sample_row, row in zip(sample_query, sample_table):
                row = [
                    str(sample_row[0].observation),
                    str(sample_row[0].point.latitude),
                    str(sample_row[0].point.longitude)] + row
                clean_row = [x if x is not None else 'None' for x in row]
                writer.writerow(clean_row)

            sample_table_io.seek(0)
            zf.writestr(f'site_data_{task_id}.csv', sample_table_io.getvalue())

            study_table_io = StringIO()
            study_table_io.write(','.join(study_covariate_display_order))
            study_table_io.write('\n')
            for row in study_table:
                clean_row = [_wrap_in_quotes_if_needed(x) if x is not None else 'None' for x in row]
                study_table_io.write(','.join(clean_row))
                study_table_io.write('\n')

            study_table_io.seek(0)  # Move to the start of the StringIO object
            # Add the CSV content to the ZIP file
            zf.writestr(f'study_data_{task_id}.csv', study_table_io.getvalue())

        LOGGER.debug(f'{zipfile_path} is created')
        # delete it after an hour
        delete_file.apply_async(args=[zipfile_path], countdown=3600)
        return f"File {zipfile_path} has been created."
    except Exception:
        LOGGER.exception('error on _prep_download')
    finally:
        session.close()


def _wrap_in_quotes_if_needed(value):
    if isinstance(value, str):
        if '"' in value:
            value = value.replace('"', '""')
        if ',' in value or '"' in value:
            return f'"{value}"'
    return value


MAX_EO_POINT_SAMPLES = 5000


def point_table_to_point_batch(csv_file):
    point_table = pd.read_csv(csv_file)
    point_features_by_year = collections.defaultdict(list)
    point_unique_id_per_year = collections.defaultdict(list)
    for index, row in point_table.iterrows():
        year = int(row[YEAR])
        point_features_by_year[year].append(
            ee.Feature(ee.Geometry.Point(
                [row[LONGITUDE], row[LATITUDE]], 'EPSG:4326'),
                {UNIQUE_ID: index}))
        point_unique_id_per_year[year].append(index)
    return point_features_by_year, point_unique_id_per_year, point_table


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
        point_features_by_year, point_unique_id_per_year, point_table = point_table_to_point_batch(csv_file)
        dataset_id, band_name = request.form.get('data_source').split(':')

        num_years_avg = int(request.form.get('num_years_avg'))
        seasonality_aggregation_fn = request.form.get('seasonality_aggregation_fn')
        julian_start_day = int(request.form.get('julian_start_day'))
        julian_end_day = int(request.form.get('julian_end_day'))
        spatial_aggregation = request.form.get('spatial_aggregation_fn')
        spatial_radius = int(request.form.get('spatial_radius'))

        sp_tm_agg_op_str = (
            f'spatial_{spatial_aggregation}({spatial_radius};'
            f'years_mean(-{num_years_avg},0;'
            f'julian_{seasonality_aggregation_fn}({julian_start_day},{julian_end_day})))')
        LOGGER.debug(f'this is the operation: {sp_tm_agg_op_str}')
        sp_tm_agg_op = parse_spatiotemporal_fn(sp_tm_agg_op_str)
        LOGGER.info(f'{dataset_id} - {band_name} - {sp_tm_agg_op}')
        point_id_value_list = gee_database_point_sampler.process_gee_dataset(
            dataset_id,
            band_name,
            point_features_by_year,
            point_unique_id_per_year,
            None,
            sp_tm_agg_op)

        value_dict = dict(point_id_value_list)
        header_id = f'{dataset_id}:{band_name} -> {sp_tm_agg_op}'
        point_table[header_id] = point_table.index.map(value_dict)
        point_table[header_id] = point_table[header_id].apply(
            lambda x: pd.to_numeric(x, errors='ignore'))

        csv_output = StringIO()
        point_table.to_csv(csv_output, index=False)
        csv_output.seek(0)
        print(csv_output.getvalue())
        return send_file(
            BytesIO(('\ufeff' + csv_output.getvalue()).encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'remote_sensed_point_table_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv')

    data_sources = [
        'ECMWF/ERA5/MONTHLY:dewpoint_2m_temperature',
        'ECMWF/ERA5/MONTHLY:maximum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:mean_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:minimum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:total_precipitation',
        'MODIS/006/MCD12Q2:EVI_Amplitude_1',
        'MODIS/006/MCD12Q2:EVI_Area_1',
        'MODIS/006/MCD12Q2:Dormancy_1',
        'MODIS/006/MCD12Q2:Greenup_1',
        'MODIS/006/MCD12Q2:Peak_1',
        'CSP/ERGo/1_0/Global/SRTM_topoDiversity:constant'
    ]
    aggregation_functions = [
        'years_mean(-2, 0; julian_max(1, 365))',
        'years_mean(-2, 0; julian_max(121, 273))',
        'years_mean(-2, 0; julian_mean(1, 365))',
        'years_mean(-2, 0; julian_mean(121, 273))',
        'years_mean(-2, 0; julian_min(1, 365))',
        'years_mean(-2, 0; julian_min(121, 273))',
        'years_mean(-5, 0; julian_max(1, 365))',
        'years_mean(-5, 0; julian_max(121, 273))',
        'years_mean(-5, 0; julian_mean(1, 365))',
        'years_mean(-5, 0; julian_mean(121, 273))',
        'years_mean(-5, 0; julian_min(1, 365))',
        'years_mean(-5, 0; julian_min(121, 273))',
    ]

    return render_template(
        'remote_sensed_data_extractor.html',
        latitude_id=LATITUDE,
        longitude_id=LONGITUDE,
        year_id=YEAR,
        max_eo_points=MAX_EO_POINT_SAMPLES,
        data_sources=data_sources,
        aggregation_functions=aggregation_functions,
    )


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

if __name__ == '__main__':
    app.run(debug=False)
