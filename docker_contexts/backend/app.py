import configparser
import logging
import os
import pickle
import re
import sys

from database import SessionLocal
from database_model_definitions import Study, Sample, Point, Species
from flask import Flask
from flask import render_template
from flask import request
from sqlalchemy import distinct, func
from sqlalchemy.engine import Row
from sqlalchemy.sql import and_, or_


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


UNIQUE_FIELD_VALUES_PKL = 'unique_field_values.pkl'


def init_unique_fieldnames():
    global UNIQUE_FIELD_VALUES
    UNIQUE_FIELD_VALUES = {}

    if os.path.exists(UNIQUE_FIELD_VALUES_PKL):
        # Load data from the pickle file
        with open(UNIQUE_FIELD_VALUES_PKL, 'rb') as f:
            UNIQUE_FIELD_VALUES = pickle.load(f)
        LOGGER.info("Loaded unique field values from pickle file.")
        return

    session = SessionLocal()
    for column in SEARCH_BY_UNIQUE_VAL:
        LOGGER.debug(f'processing {column}')
        LOGGER.debug(f'i.e. {column.name}')
        unique_values = session.query(distinct(column)).all()
        UNIQUE_FIELD_VALUES[column.name] = [v[0] for v in unique_values]

    session.close()

    with open(UNIQUE_FIELD_VALUES_PKL, 'wb') as f:
        pickle.dump(UNIQUE_FIELD_VALUES, f)
    LOGGER.info(f"Saved unique field values to {UNIQUE_FIELD_VALUES_PKL}.")


init_unique_fieldnames()


@app.route('/api/')
def index():
    session = SessionLocal()
    # Example: Fetch all records from ExampleModel
    samples = session.query(Sample).all()
    return 'Number of samples: ' + str(len(samples))


OPERATION_MAP = {
    '=': lambda field, value: field == value,
    '<': lambda field, value: field < value,
    '>': lambda field, value: field > value
}


@app.route('/api/home')
def home():
    session = SessionLocal()
    n_samples = session.query(Sample).count()

    response_variables = session.query(distinct(Sample.response_variable)).filter(
        Sample.response_variable.isnot(None),
        Sample.response_variable != ''
    ).all()
    response_variables = [value[0] for value in response_variables]

    response_types = session.query(distinct(Sample.response_type)).filter(
        Sample.response_type.isnot(None),
        Sample.response_type != ''
    ).all()
    response_types = [value[0] for value in response_types]

    country_set = session.query(distinct(Point.country)).all()
    country_set = [value[0] for value in country_set]
    continent_set = session.query(distinct(Point.continent)).all()
    continent_set = [value[0] for value in continent_set]

    return render_template(
        'query_builder.html',
        status_message=f'Number of samples in db: {n_samples}',
        possible_operations=list(OPERATION_MAP),
        unique_fields=UNIQUE_FIELD_VALUES.keys(),
        response_variables=response_variables,
        response_types=response_types,
        country_set=country_set,
        continent_set=continent_set,
        unique_field_values=UNIQUE_FIELD_VALUES,
        )


def to_dict(obj):
    """
    Convert a SQLAlchemy model object into a dictionary.
    """
    if isinstance(obj, Row):
        return obj._mapping
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        fields = request.form.getlist('field')
        operations = request.form.getlist('operation')
        values = request.form.getlist('value')

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
                m = re.match(r"[(]?([^, ]+)[, ]*([^, )]+)[\)]?", center_point)
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
            filters.append(Point.country == country_select)
        continent_select = request.form.get('continentSelect')
        if continent_select:
            filters.append(Point.continent == continent_select)

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
                    Sample.study_id
                ).filter(Sample.response_variable == min_observations_response_variable).group_by(
                Sample.study_id, Sample.response_variable).having(
                func.count(Sample.id_key) >= min_observations)
            valid_study_ids = [
                row[0] for row in studies_with_min_observations.all()]
            filters.append(
                Sample.study_id.in_(valid_study_ids))

        min_site_years = request.form.get('minSiteYears')
        if min_site_years:
            unique_years_count_query = session.query(
                Sample.study_id).group_by(
                Sample.study_id).having(
                    func.count(distinct(
                        Sample.year + '_' + Sample.point_id)) >= int(
                        min_site_years))

            valid_study_ids = [
                row[0] for row in unique_years_count_query.all()]
            filters.append(Sample.study_id.in_(valid_study_ids))

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
        LOGGER.debug(f'***** {sample_size_min_years}')
        if sample_size_min_years:
            unique_years_count_query = session.query(
                Sample.study_id).group_by(
                Sample.study_id).having(
                    func.count(distinct(Sample.year)) >= int(sample_size_min_years))
            valid_study_ids = [
                row[0] for row in unique_years_count_query.all()]
            LOGGER.debug(f'********* valid study ids: {valid_study_ids}')
            filters.append(Sample.study_id.in_(valid_study_ids))

        min_observations_per_year = request.form.get('sampleSizeMinObservationsPerYear')
        if min_observations_per_year:
            unique_years_count_query = session.query(
                Sample.study_id,
                func.count(Sample.study_id).label('sample_count')).group_by(
                Sample.study_id, Sample.year).subquery()
            min_samples_per_study_query = session.query(
                unique_years_count_query.c.study_id).group_by(
                unique_years_count_query.c.study_id).having(
                func.min(unique_years_count_query.c.sample_count) >=
                int(min_observations_per_year))
            valid_study_ids = [
                row[0] for row in min_samples_per_study_query.all()]
            filters.append(Sample.study_id.in_(valid_study_ids))

        study_query = session.query(Study).join(
            Sample, Sample.study_id == Study.id_key).join(
            Point, Sample.point_id == Point.id_key).filter(
            and_(*filters))

        # determine what response types are in this query
        filtered_response_types = session.query(
            distinct(Sample.response_type)).join(
            Study, Sample.study_id == Study.id_key).join(
            Point, Sample.point_id == Point.id_key).filter(
            and_(*filters)
        ).filter(Sample.response_type.isnot(None)).all()

        fields_to_select = SAMPLE_DISPLAY_FIELDS.copy()
        for response_type, fields in FIELDS_BY_RESPONSE_TYPE.items():
            LOGGER.debug(f'testing if {response_type} is in {filtered_response_types}')
            if response_type.lower() in [
                    rt[0].lower() for rt in filtered_response_types]:
                LOGGER.debug(f'because of {filtered_response_types} extending these field {fields}')
                fields_to_select.extend(fields)

        LOGGER.debug(f'about to query the samples, here are the filters: {filters}')
        sample_query = session.query(*fields_to_select).join(
            Study, Sample.study_id == Study.id_key).join(
            Point, Sample.point_id == Point.id_key).filter(
            and_(*filters))

        study_query_result = [to_dict(s) for s in study_query.all()]
        sample_query_result = [to_dict(s) for s in sample_query.all()]

        key_sets = [set(d.keys()) for d in sample_query_result]
        all_keys = set().union(*key_sets)
        common_keys = set(key_sets[0]).intersection(*key_sets[1:])
        not_in_all_keys = all_keys - common_keys
        LOGGER.debug(f'these are the common keys: {common_keys}')
        LOGGER.debug(f'these are the disjoint keys: {not_in_all_keys}')

        point_set = set([
            (s['latitude'], s['longitude'])
            for s in sample_query_result])
        points = [
            {"lat": s[0], "lng": s[1]}
            for s in point_set]
        LOGGER.debug(
            f'samples: {len(sample_query_result)} points: {len(points)}')
        LOGGER.debug(f'keys: {sample_query_result[0]}')
        LOGGER.debug(f'keys: {sample_query_result[1]}')

        session.close()
        return render_template(
            'results_view.html',
            studies=study_query_result,
            samples=sample_query_result,
            points=points)
    except Exception as e:
        LOGGER.exception(f'error with {e}')
        raise


from forms import SpeciesForm
from flask import redirect, url_for


@app.route('/admin/species', methods=['GET', 'POST'])
def admin_species():
    session = SessionLocal()
    form = SpeciesForm()
    if form.validate_on_submit():
        name = form.name.data
        new_species = Species(name=name)
        session.add(new_species)
        session.commit()
        return redirect(url_for('admin_species'))
    species_list = session.query(Species).all()
    return render_template('admin_species.html', form=form, species_list=species_list)


@app.route('/admin/species/delete/<int:id_key>', methods=['POST'])
def delete_species(id_key):
    session = SessionLocal()
    species = session.query(Species).get(id_key)
    session.delete(species)
    session.commit()
    return redirect(url_for('admin_species'))


if __name__ == '__main__':
    app.run(debug=True)
