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
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, GeolocationName
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


OPERATION_MAP = {
    '=': lambda field, value: field == value,
    '<': lambda field, value: field < value,
    '>': lambda field, value: field > value
}


def to_dict(covariate_list):
    covariate_dict = collections.defaultdict(lambda: None)
    for covariate in covariate_list:
        covariate_dict[covariate.covariate_defn.name] = covariate.value
    return covariate_dict


def calculate_covariate_display_order(session, query_to_filter, covariate_type):
    pre_covariate_display_order = [x for x in (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .order_by(CovariateDefn.display_order,
                  func.lower(CovariateDefn.name))
        .filter(
            CovariateDefn.covariate_association ==
            covariate_type)
    ).all()]

    unique_values_per_covariate = collections.defaultdict(set)
    for index, row in enumerate(query_to_filter):
        for covariate in row.covariates:
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
    for row in query_to_filter:
        covariate_dict = to_dict(row.covariates)
        display_table.append([
            covariate_dict[name]
            for name in covariate_display_order])
    return covariate_display_order, display_table


@app.route('/pcld')
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
            distinct(GeolocationName.geolocation_name)).filter(
            GeolocationName.geolocation_type == 'COUNTRY').all()]
    continent_set = [
        x[0] for x in session.query(
            distinct(GeolocationName.geolocation_name)).filter(
            GeolocationName.geolocation_type == 'CONTINENT').all()]

    return render_template(
        'query_builder.html',
        status_message=f'Number of samples in db: {n_samples}',
        possible_operations=list(OPERATION_MAP),
        country_set=country_set,
        continent_set=continent_set,
        unique_covariate_values=searchable_covariates,
        )


@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        fields = request.form.getlist('field')
        operations = request.form.getlist('operation')
        values = request.form.getlist('value')
        max_sample_size = request.form.get('max_sample_size')

        # Example of how you might process these queries
        filters = []
        for field, operation, value in zip(fields, operations, values):
            if not field:
                continue
            if hasattr(Study, field):
                column = getattr(Study, field)
            elif hasattr(Sample, field):
                column = getattr(Sample, field)
            else:
                raise AttributeError(f"Field '{field}' not found in Study or Sample.")
            if operation == '=' and '*' in value:
                value = value.replace('*', '%')
            filter_condition = OPERATION_MAP[operation](column, value)
            filters.append(filter_condition)
        session = SessionLocal()

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

        upper_left_point = request.form.get('upperLeft')
        if upper_left_point is not None:
            upper_left_point = request.form.get('upperLeft').strip()
            lower_right_point = request.form.get('lowerRight').strip()

            if upper_left_point != '':
                m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", upper_left_point)
                ul_lat, ul_lng = [float(v) for v in m.group(1, 2)]
                m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", lower_right_point)
                lr_lat, lr_lng = [float(v) for v in m.group(1, 2)]

        country_select = request.form.get('countrySelect')
        if country_select:
            geolocation_subquery = (
                session.query(Point.id_key)
                .join(Point.geolocations)
                .filter(GeolocationName.geolocation_name == country_select)
            ).subquery()
            filters.append(Point.id_key.in_(select(geolocation_subquery)))

        continent_select = request.form.get('continentSelect')
        if continent_select:
            geolocation_subquery = (
                session.query(Point.id_key)
                .join(Point.geolocations)
                .filter(GeolocationName.geolocation_name == continent_select)
            ).subquery()
            filters.append(Point.id_key.in_(geolocation_subquery))

        if ul_lat is not None:
            filters.append(Point.latitude <= ul_lat)
            filters.append(Point.latitude >= lr_lat)
            filters.append(Point.longitude <= ul_lng)
            filters.append(Point.longitude >= lr_lng)

        year_range = request.form.get('yearRange')
        if year_range is not None:
            year_range = year_range.strip()
            if year_range != '':
                m = [
                    v.group() for v in
                    re.finditer(r"(\d+ ?- ?\d+)|(?:(\d+))", year_range)]
                year_filters = []
                for year_group in m:
                    if '-' not in year_group:
                        year = int(year_group.strip())
                        year_filters.append(Sample.year == year)
                    else:
                        start_year, end_year = [
                            int(v.strip()) for v in year_group.split('-')]
                        year_filters.append(
                            and_(
                                Sample.year >= start_year,
                                Sample.year <= end_year))
                filters.append(or_(*year_filters))

        # add filters for the min observation side
        min_observations_response_variable = request.form.get(
            'minObservationsResponseVariable')

        if min_observations_response_variable:
            min_observations = int(request.form.get('minObservationsSampleSize'))
            studies_with_min_observations = session.query(
                    Sample.study_id).filter(Sample.response_variable == min_observations_response_variable).group_by(
                Sample.study_id, Sample.response_variable).having(
                func.count(Sample.id_key) >= min_observations)
            valid_study_ids = [
                row[0] for row in studies_with_min_observations.all()]
            filters.append(
                Sample.study_id.in_(valid_study_ids))

        min_site_years = request.form.get('minSiteYears')
        if min_site_years:
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
                if row.unique_years >= int(min_site_years)]
            filters.append(
                tuple_(Sample.study_id, Sample.point_id).in_(valid_sites))

        min_sites_response_type = request.form.get('minSitesResponseType')
        if min_sites_response_type:
            min_sites_per_response_type_count = int(
                request.form.get('minSitesResponseTypeCount'))
            min_sites = session.query(
                Sample.study_id).filter(
                Sample.response_type == min_sites_response_type).group_by(
                Sample.study_id).having(func.count(
                    distinct(Sample.point_id)) >= min_sites_per_response_type_count)
            valid_study_ids = [row[0] for row in min_sites.all()]
            filters.append(Sample.study_id.in_(valid_study_ids))

        sample_size_min_years = request.form.get('sampleSizeMinYears')
        if sample_size_min_years:
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
                if row[1] >= int(sample_size_min_years)]
            filters.append(Sample.study_id.in_(valid_study_ids))

        min_observations_per_year = request.form.get('sampleSizeMinObservationsPerYear')
        if min_observations_per_year:
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

        LOGGER.info(f'about to query on samples {filters}')
        sample_query = (
            session.query(Sample)
            .options(subqueryload(Sample.covariates))
            .join(Point, Sample.point_id == Point.id_key)
            .filter(*filters)
            .limit(max_sample_size)
        )

        LOGGER.info(f'about to query on study {filters}')
        study_query = (
            session.query(Study)
            .join(Sample, Study.id_key == Sample.study_id)
            .join(Point, Sample.point_id == Point.id_key)
            .filter(*filters)
        )
        # determine what covariate ids are in this query
        LOGGER.info('determine study covariates to display')
        study_covariate_ids_to_display = set(REQUIRED_STUDY_FIELDS)
        for study in study_query:
            study_covariate_ids_to_display |= set([c.covariate_defn.name for c in study.covariates])

        LOGGER.info('calculate covariate display order for study')
        study_covariate_display_order, study_table = calculate_covariate_display_order(
            session, study_query, CovariateAssociation.STUDY)
        LOGGER.info('calculate covariate display order for sample')
        sample_covariate_display_order, sample_table = calculate_covariate_display_order(
            session, sample_query, CovariateAssociation.SAMPLE)

        # add the lat/lng points and observation to the sample
        sample_covariate_display_order = [
            OBSERVATION, LATITUDE, LONGITUDE] + sample_covariate_display_order
        for sample_row, display_row in zip(sample_query, sample_table):
            display_row[:] = [
                sample_row.observation,
                sample_row.point.latitude,
                sample_row.point.longitude] + display_row

        LOGGER.info(f'about to query on points')
        point_query = (
            session.query(Point)
            .join(Sample, Point.id_key == Sample.point_id)
            .filter(Sample.id_key.in_([s.id_key for s in sample_query]))
            .distinct()
        ).all()

        points = [
            {"lat": p.latitude, "lng": p.longitude}
            for p in point_query]

        session.close()
        return render_template(
            'results_view.html',
            study_headers=study_covariate_display_order,
            study_table=study_table,
            sample_headers=sample_covariate_display_order,
            sample_table=sample_table,
            points=points
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
