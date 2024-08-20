from io import StringIO
from io import TextIOWrapper
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

import gee_dataset_point_sampler
import numpy
from database import SessionLocal
from database_model_definitions import REQUIRED_STUDY_FIELDS, REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import OBSERVATION, LATITUDE, LONGITUDE, YEAR
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import send_file
from flask import flash
from flask import redirect
from flask import make_response
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
from celery_config import make_celery
from celery import current_task
from celery.signals import after_setup_logger
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


INSTANCE_DIR = './instance'
QUERY_RESULTS_DIR = os.path.join(INSTANCE_DIR, 'results_to_download')
os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)
MAX_SAMPLE_DISPLAY_SIZE = 1000


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


def calculate_sample_display_table(session, query_to_filter):
    pre_covariate_display_query = (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .filter(
            or_(CovariateDefn.covariate_association == CovariateAssociation.SAMPLE.value,
                CovariateDefn.show_in_point_table == 1))
        .order_by(
            CovariateDefn.display_order,
            func.lower(CovariateDefn.name)))

    unique_values_per_covariate = collections.defaultdict(set)
    sample_covariate_list = []
    for index, (sample, study) in enumerate(query_to_filter):
        sample_covariates = sample.covariates
        study_covariates = [
            cov for cov in study.covariates
            if cov.covariate_defn.show_in_point_table == 1
        ]
        all_covariates = sample_covariates + study_covariates
        sample_covariate_list.append((sample, all_covariates))

        for covariate in all_covariates:
            if not isinstance(covariate.value, str) and (
                    covariate.value is None or numpy.isnan(covariate.value)):
                continue
            if isinstance(covariate.value, str):
                if covariate.value == 'null':
                    continue
                try:
                    # see if it could be numeric, if so just put true
                    float(covariate.value)
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(True)
                except ValueError:
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(
                            covariate.value.lower())
            else:
                # it's a numeric, just note it's defined
                unique_values_per_covariate[
                    covariate.covariate_defn.name].add(True)

    # get all possible conditions
    covariate_display_order = []

    for name, always_display, hidden, condition in pre_covariate_display_query:
        if hidden:
            continue
        if condition is None or condition == 'null':
            if always_display or unique_values_per_covariate[name]:
                covariate_display_order.append(name)

        elif condition['value'].lower() in unique_values_per_covariate[
                condition['depends_on']]:
            covariate_display_order.append(name)

    display_table = []
    for sample, covariate_list in sample_covariate_list:
        covariate_dict = to_dict(covariate_list)
        display_table.append([
            covariate_dict[name]
            for name in covariate_display_order])
    return covariate_display_order, display_table


def calculate_study_display_order(
        session, query_to_filter):
    pre_covariate_display_order = (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .filter(
            or_(CovariateDefn.covariate_association == CovariateAssociation.STUDY.value,
                CovariateDefn.show_in_point_table == 1))
        .order_by(
            CovariateDefn.display_order,
            func.lower(CovariateDefn.name))
        )

    unique_values_per_covariate = collections.defaultdict(set)
    for index, study in enumerate(query_to_filter):
        for covariate in study.covariates:
            if not isinstance(covariate.value, str) and (
                    covariate.value is None or numpy.isnan(covariate.value)):
                continue
            if isinstance(covariate.value, str):
                if covariate.value == 'null':
                    continue
                try:
                    # see if it could be numeric, if so just put true
                    float(covariate.value)
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(True)
                except ValueError:
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(
                            covariate.value.lower())
            else:
                # it's a numeric, just note it's defined
                unique_values_per_covariate[
                    covariate.covariate_defn.name].add(True)

    # get all possible conditions
    covariate_display_order = []

    for name, always_display, hidden, condition in pre_covariate_display_order:
        if hidden:
            continue
        if condition is None or condition == 'null':
            if always_display or unique_values_per_covariate[name]:
                covariate_display_order.append(name)

        elif condition['value'].lower() in unique_values_per_covariate[
                condition['depends_on']]:
            covariate_display_order.append(name)

    display_table = []
    for study in query_to_filter:
        covariate_dict = to_dict(study.covariates)
        display_table.append([
            covariate_dict[name]
            for name in covariate_display_order])
    return covariate_display_order, display_table


def initialize_searchable_covariates():
    global COVARIATE_STATE
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    pkcl_filepath = os.path.join(INSTANCE_DIR, 'initialize_searchable_covariates.pkl')
    if os.path.exists(pkcl_filepath):
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

    with open(pkcl_filepath, 'wb') as file:
        pickle.dump(COVARIATE_STATE, file)
    LOGGER.info('all done with unique values')


@app.route('/')
def pcld():
    global COVARIATE_STATE
    return render_template(
        'query_builder.html',
        status_message=f'Number of samples in db: {n_samples}',
        possible_operations=list(OPERATION_MAP),
        country_set=COVARIATE_STATE['country_set'],
        continent_set=COVARIATE_STATE['continent_set'],
        unique_covariate_values=COVARIATE_STATE['searchable_covariates'],
        )


@app.route('/api/n_samples', methods=['POST'])
def n_samples():
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
        .join(Sample.study)
        .join(Sample.point)
        .filter(
            *filters
        ).distinct()
    )
    study_count = study_query.count()
    session.close()

    return jsonify({
        'sample_count': sample_count,
        'study_count': study_count,
        'filter_text': filter_text
        })


def form_to_dict(form):
    centerPointBuffer=form.get('centerPointBuffer')

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
    for field, operation, value in zip(fields, operations, values):
        if not field or not value:
            continue
        covariate_type = session.query(
            CovariateDefn.covariate_association).filter(
            CovariateDefn.name == field).first()[0]
        filter_text += f'{field}({covariate_type}) {operation} {value}\n'

        if covariate_type == CovariateAssociation.STUDY.value:
            covariate_subquery = (
                session.query(CovariateValue.study_id)
                .join(CovariateValue.covariate_defn)
                .filter(
                    and_(CovariateDefn.name == field,
                         OPERATION_MAP[operation](CovariateValue.value, value))
                ).subquery())
            filters.append(
                Study.id_key.in_(session.query(covariate_subquery.c.study_id)))

        elif covariate_type == CovariateAssociation.SAMPLE.value:
            covariate_subquery = (
                session.query(CovariateValue.sample_id)
                .join(CovariateValue.covariate_defn)
                .filter(
                    and_(CovariateDefn.name == field,
                         OPERATION_MAP[operation](CovariateValue.value, value))
                ).subquery())
            filters.append(
                Sample.id_key.in_(session.query(covariate_subquery.c.sample_id)))

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
            ul_lng = lng+center_point_buffer/2
            lr_lng = lng-center_point_buffer/2

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
        filters.append(Point.id_key.in_(geolocation_subquery))
        filter_text += f'continent is {continent_select}\n'

    if ul_lat is not None:
        filters.append(and_(
            Point.latitude <= ul_lat,
            Point.latitude >= lr_lat,
            Point.longitude <= ul_lng,
            Point.longitude >= lr_lng))

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
        unique_years_count_query = (
            session.query(
                Sample.study_id,
                func.count(func.distinct(CovariateValue.value)).label('unique_years')
            ).join(Sample.covariates)
             .join(CovariateValue.covariate_defn)
             .filter(CovariateDefn.name == 'year')
             .group_by(Sample.study_id))
        valid_study_ids = [
            row[0] for row in unique_years_count_query.all()
            if row[1] >= sample_size_min_years]
        filters.append(Sample.study_id.in_(valid_study_ids))

    min_observations_per_year = int(form['sampleSizeMinObservationsPerYear'])
    if min_observations_per_year > 0:
        filter_text += f'min observations per year {min_observations_per_year}\n'
        query = (
            session.query(
                Sample.study_id,
                CovariateValue.value,
                func.count(Sample.study_id).label('count_per_year')
            )
            .join(Sample.covariates)
            .join(CovariateValue.covariate_defn)
            .filter(CovariateDefn.name == 'year')
            .group_by(Sample.study_id, CovariateValue.value)
        )
        study_obs_per_year = {}
        for study_id, year, samples_per_year in query:
            if study_id not in study_obs_per_year or samples_per_year < study_obs_per_year[study_id][1]:
                study_obs_per_year[study_id] = (year, samples_per_year)
        valid_study_ids = [
            study_id for study_id, (year, samples_per_year)
            in study_obs_per_year.items()
            if samples_per_year >= int(min_observations_per_year)]
        filters.append(Sample.study_id.in_(valid_study_ids))

    year_range = form['yearRange']
    if year_range:
        filter_text += 'years in {' + year_range + '}\n'
        year_set = extract_years(year_range)
        year_subquery = (
            session.query(CovariateValue.sample_id)
            .join(CovariateValue.covariate_defn)
            .filter(
                and_(CovariateDefn.name == 'year',
                     CovariateValue.value.in_(year_set))
            ).subquery())
        filters.append(
            Sample.id_key.in_(session.query(year_subquery.c.sample_id)))
    return filters, filter_text


def explain_query(session, query):
    compiled_query = query.statement.compile(session.bind, compile_kwargs={"literal_binds": True})
    explain_query = f"EXPLAIN QUERY PLAN {compiled_query}"
    result = session.execute(text(explain_query)).fetchall()
    for row in result:
        print(row)


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

        sample_covariate_display_order, sample_table = calculate_sample_display_table(
            session, sample_query.limit(MAX_SAMPLE_DISPLAY_SIZE))

        study_query = (
            session.query(Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .options(selectinload(Study.covariates))
        )
        unique_studies = {study for study in study_query}

        # determine what covariate ids are in this query
        study_covariate_display_order, study_table = calculate_study_display_order(
            session, unique_studies)

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
        covariate_association_states=[x.value for x in CovariateAssociation],)


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
        'condition': c.condition,
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
        if remote_covariate['condition'] != "None":
            local_covariate.condition = remote_covariate['condition']
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
        filters, filter_text = build_filter(session, query_form)

        sample_query = (
            session.query(Sample, Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .options(selectinload(Sample.covariates))
        ).yield_per(1000).limit(50000)

        study_query = (
            session.query(Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .options(selectinload(Study.covariates))
        )

        unique_studies = {study for study in study_query}
        # determine what covariate ids are in this query
        LOGGER.info('calculate covariate display order for study')
        study_covariate_display_order, study_table = calculate_study_display_order(
            session, unique_studies)
        sample_covariate_display_order, sample_table = calculate_sample_display_table(
            session, sample_query)
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
                    str(sample_row[0].point.longitude),] + row
                clean_row = [x if x is not None else 'None' for x in row]
                writer.writerow(clean_row)

            sample_table_io.seek(0)
            zf.writestr(f'site_data_{task_id}.csv', sample_table_io.getvalue())

            study_table_io = StringIO()
            study_table_io.write(','.join(study_covariate_display_order))
            study_table_io.write('\n')
            for row in study_table:
                clean_row = [x if x is not None else 'None' for x in row]
                study_table_io.write(','.join(clean_row))
                study_table_io.write('\n')

            study_table_io.seek(0)  # Move to the start of the StringIO object
            # Add the CSV content to the ZIP file
            zf.writestr(f'study_data_{task_id}.csv', study_table_io.getvalue())

        LOGGER.debug(f'{zipfile_path} is created')
        return f"File {zipfile_path} has been created."
    except Exception:
        LOGGER.exception('error on _prep_download')
    finally:
        session.close()


MAX_EO_POINT_SAMPLES = 100


@app.route('/data_extractor', methods=['GET', 'POST'])
def data_extractor():
    if request.method == 'POST':
        # Handle form data
        # dropdown1 = request.form.get('dropdown1')
        # textbox1 = request.form.get('textbox1')
        csv_file = request.files.get('csv_file')
        gee_dataset_point_sampler(csv_file)

        flash('File uploaded and validated successfully!', 'success')
        return redirect(url_for('data_extractor'))

    data_sources = [
        'ECMWF/ERA5/MONTHLY:dewpoint_2m_temperature',
        'ECMWF/ERA5/MONTHLY:maximum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:mean_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:minimum_2m_air_temperature',
        'ECMWF/ERA5/MONTHLY:total_precipitation',
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
        if not csv_data:
            invalid_message += 'Data in file!<br/>'
            flash('Data in file!', 'danger')
        else:
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
                header = next(csv_reader)  # Try to read the header
            except csv.Error as e:
                invalid_message += f'The file is not a valid CSV ({str(e)})!\n'
        if invalid_message:
            flash(invalid_message, 'danger')
        else:
            flash('valid csv', 'info')
        return jsonify({
            'valid': False if invalid_message else True,
            'message': invalid_message})
    except Exception as e:
        flash(str(e), 'danger')
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
    initialize_searchable_covariates()

if __name__ == '__main__':
    app.run(debug=True)
