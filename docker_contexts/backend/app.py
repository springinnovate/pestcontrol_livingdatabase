import configparser
import collections
import logging
import os
import pickle
import re
import sys

import numpy
from database import SessionLocal
from database_model_definitions import REQUIRED_STUDY_FIELDS, REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import OBSERVATION, LATITUDE, LONGITUDE
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from sqlalchemy import select
from sqlalchemy import distinct, func
from sqlalchemy.engine import Row
from sqlalchemy.sql import and_, or_, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import String
from sqlalchemy.orm import subqueryload
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import joinedload, contains_eager

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


UNIQUE_COVARIATE_VALUES_PKL = 'unique_COVARIATE_values.pkl'


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
    pre_covariate_display_order = [x for x in (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .filter(
            or_(CovariateDefn.covariate_association == CovariateAssociation.SAMPLE,
                CovariateDefn.show_in_point_table == 1))
        .order_by(
            CovariateDefn.display_order,
            func.lower(CovariateDefn.name))).all()]

    # now filter the covariates by what is actually defined

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
    for sample, covariate_list in sample_covariate_list:
        covariate_dict = to_dict(covariate_list)
        display_table.append([
            covariate_dict[name]
            for name in covariate_display_order])
    return covariate_display_order, display_table


def calculate_study_display_order(
        session, query_to_filter):
    pre_covariate_display_order = [x for x in (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .filter(
            or_(CovariateDefn.covariate_association == CovariateAssociation.STUDY,
                CovariateDefn.show_in_point_table == 1))
        .order_by(
            CovariateDefn.display_order,
            func.lower(CovariateDefn.name))).all()]

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


@app.route('/')
def pcld():
    session = SessionLocal()

    searchable_unique_covariates = (
        session.query(
            CovariateDefn.name,
            group_concat_distinct(CovariateValue.value).label('unique_values')
        ).filter(
            CovariateDefn.queryable,
            CovariateDefn.covariate_type == CovariateType.STRING
            )
        .join(CovariateValue, CovariateDefn.id_key == CovariateValue.covariate_defn_id)
        .group_by(CovariateDefn.name)
    ).all()
    searchable_covariates = {
        row.name: row.unique_values.split(',')
        for row in searchable_unique_covariates if row.unique_values}

    searchable_continuous_covariates = (
        session.query(
            CovariateDefn.name, CovariateDefn.covariate_type
        ).filter(
            CovariateDefn.queryable,
            CovariateDefn.covariate_type != CovariateType.STRING
        )
    ).all()

    searchable_covariates.update({
        row.name: str(row.covariate_type)
        for row in searchable_continuous_covariates})

    n_samples = session.query(Sample).count()

    country_set = [
        x[0] for x in session.query(
            distinct(Geolocation.geolocation_name)).filter(
            Geolocation.geolocation_type == 'COUNTRY').all()]
    continent_set = [
        x[0] for x in session.query(
            distinct(Geolocation.geolocation_name)).filter(
            Geolocation.geolocation_type == 'CONTINENT').all()]

    return render_template(
        'query_builder.html',
        status_message=f'Number of samples in db: {n_samples}',
        possible_operations=list(OPERATION_MAP),
        country_set=country_set,
        continent_set=continent_set,
        unique_covariate_values=searchable_covariates,
        )


@app.route('/api/n_samples', methods=['POST'])
def n_samples():
    session = SessionLocal()
    sample_count = session.query(func.count(Sample.id_key)).scalar()
    study_count = session.query(func.count(Study.id_key)).scalar()
    covariate_count = session.query(func.count(CovariateDefn.id_key)).scalar()
    session.close()
    return jsonify({
        'sample_count': sample_count,
        'study_count': study_count,
        'covariate_count': covariate_count,
        })


@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        fields = request.form.getlist('covariate')
        operations = request.form.getlist('operation')
        values = request.form.getlist('value')
        max_sample_size = request.form.get('maxSampleSize')

        session = SessionLocal()
        filters = []

        filter_text = ''
        for field, operation, value in zip(fields, operations, values):
            if not field:
                continue
            covariate_type = session.query(
                CovariateDefn.covariate_association).filter(
                CovariateDefn.name == field).first()[0]
            filter_text += f'{field}({covariate_type}) {operation} {value}\n'

            if covariate_type == CovariateAssociation.STUDY:
                covariate_subquery = (
                    session.query(CovariateValue.study_id)
                    .join(CovariateValue.covariate_defn)
                    .filter(
                        and_(CovariateDefn.name == field,
                             OPERATION_MAP[operation](CovariateValue.value, value))
                    ).subquery())
                filters.append(
                    Study.id_key.in_(session.query(covariate_subquery.c.study_id)))

            elif covariate_type == CovariateAssociation.SAMPLE:
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
        center_point = request.form.get('centerPoint')
        if center_point is not None:
            center_point = center_point.strip()
            if center_point != '':
                m = re.match(r"[(]?([^, \t]+)[, \t]+([^, )]+)[\)]?", center_point)
                LOGGER.debug(f'MATCH {m}')
                lat, lng = [float(v) for v in m.group(1, 2)]
                center_point_buffer = float(
                    request.form.get('centerPointBuffer').strip())/2
                ul_lat = lat+center_point_buffer/2
                lr_lat = lat-center_point_buffer/2
                ul_lng = lng+center_point_buffer/2
                lr_lng = lng-center_point_buffer/2

                filter_text += f'center point at ({lat},{lng}) + +/-{center_point_buffer}\n'

        upper_left_point = request.form.get('upperLeft')
        if upper_left_point is not None:
            upper_left_point = request.form.get('upperLeft').strip()
            lower_right_point = request.form.get('lowerRight').strip()

            if upper_left_point != '':
                m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", upper_left_point)
                ul_lat, ul_lng = [float(v) for v in m.group(1, 2)]
                m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", lower_right_point)
                lr_lat, lr_lng = [float(v) for v in m.group(1, 2)]

                filter_text += f'bounding box ({ul_lat},{ul_lng}) ({lr_lat},{lr_lng})\n'

        country_select = request.form.get('countrySelect')
        if country_select:
            geolocation_subquery = (
                session.query(Point.id_key)
                .join(Point.geolocations)
                .filter(Geolocation.geolocation_name == country_select)
            ).subquery()
            filters.append(Point.id_key.in_(select(geolocation_subquery)))
            filter_text += f'country is {country_select}\n'

        continent_select = request.form.get('continentSelect')
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

        LOGGER.debug(request.form)
        min_site_years = int(request.form.get('minSiteYears'))
        if min_site_years > 0:
            filter_text += f'min site years={min_site_years}\n'
            unique_years_count_query = (
                session.query(
                    Sample.study_id,
                    Sample.point_id,
                    func.count(func.distinct(CovariateValue.value)).label(
                        'unique_years')
                )
                .join(Sample.covariates)
                .join(CovariateValue.covariate_defn)
                .filter(CovariateDefn.name == 'year')
                .group_by(Sample.study_id, Sample.point_id)
            )

            valid_sites = [
                (row.study_id, row.point_id) for row in unique_years_count_query
                if row.unique_years >= min_site_years]
            filters.append(
                tuple_(Sample.study_id, Sample.point_id).in_(valid_sites))

        min_sites_per_study = int(request.form.get('minSitesPerStudy'))
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

        sample_size_min_years = int(request.form.get('sampleSizeMinYears'))
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

        min_observations_per_year = int(request.form.get('sampleSizeMinObservationsPerYear'))
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

        year_range = request.form.get('yearRange')
        if year_range:
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

        sample_query = (
            session.query(Sample, Study)
            .join(Sample.study)
            .join(Sample.point)
            .filter(
                *filters
            )
            .limit(max_sample_size)
        )

        LOGGER.info('calculate covariate display order for sample')
        sample_covariate_display_order, sample_table = calculate_sample_display_table(
            session, sample_query)

        unique_studies = {sample[1] for sample in sample_query}

        # determine what covariate ids are in this query
        LOGGER.info('calculate covariate display order for study')
        study_covariate_display_order, study_table = calculate_study_display_order(
            session, unique_studies)

        # add the lat/lng points and observation to the sample
        sample_covariate_display_order = [
            OBSERVATION, LATITUDE, LONGITUDE] + sample_covariate_display_order
        for (sample_row, _), display_row in zip(sample_query, sample_table):
            display_row[:] = [
                sample_row.observation,
                sample_row.point.latitude,
                sample_row.point.longitude] + display_row

        LOGGER.info('about to query on points')
        unique_points = {sample[0].point for sample in sample_query}
        points = [
            {"lat": p.latitude, "lng": p.longitude}
            for p in unique_points]

        session.close()
        LOGGER.info('all done, sending to results view...')
        return render_template(
            'results_view.html',
            study_headers=study_covariate_display_order,
            study_table=study_table,
            sample_headers=sample_covariate_display_order,
            sample_table=sample_table,
            points=points,
            compiled_query=f'<pre>Filter as text: {filter_text}</pre>'
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
        'covariate_association': c.covariate_association.value,
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


if __name__ == '__main__':
    app.run(debug=True)
