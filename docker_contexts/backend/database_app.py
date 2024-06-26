import configparser
import re
import datetime
import logging
import sys

from database import SessionLocal
from database_model_definitions import BASE_FIELDS
from database_model_definitions import Study, Sample, Covariate
from database_model_definitions import STUDY_LEVEL_VARIABLES
from database_model_definitions import FILTERABLE_FIELDS
from database_model_definitions import FIELDS_BY_REPONSE_TYPE
from database_model_definitions import COORDINATE_PRECISION_FIELD
from database_model_definitions import COORDINATE_PRECISION_FULL_PRECISION_VALUE
from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from jinja2 import Template
from sqlalchemy import inspect
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
    inspector = inspect(Study)
    study_columns = [
        column.name for column in inspector.columns
        if not column.primary_key]
    inspector = inspect(Sample)
    sample_columns = [
        column.name for column in inspector.columns
        if not column.primary_key]
    inspector = inspect(Covariate)
    covariate_columns = [
        column.name for column in inspector.columns
        if not column.primary_key]

    session = SessionLocal()
    n_samples = session.query(Sample).count()

    return render_template(
        'query_builder.html',
        status_message=f'Number of samples in db: {n_samples}',
        possible_operations=list(OPERATION_MAP),
        fields=study_columns+sample_columns+covariate_columns)


@app.route('/api/template')
def template():
    return render_template(
        'template_builder.html',
        study_level_fields=STUDY_LEVEL_VARIABLES,
        coordinate_precision_field=COORDINATE_PRECISION_FIELD,
        coordinate_precision_full_precision_value=COORDINATE_PRECISION_FULL_PRECISION_VALUE)


@app.route('/api/build_template', methods=['POST'])
def build_template():
    # Loop through the request.form dictionary
    study_level_var_list = []
    response_type_variable_list = None
    precision_level = None
    for study_level_key in STUDY_LEVEL_VARIABLES:
        if isinstance(study_level_key, tuple):
            study_level_key = study_level_key[0]
            if study_level_key == COORDINATE_PRECISION_FIELD:
                precision_level = request.form[study_level_key]
                study_level_var_list.append(
                    (study_level_key, precision_level))
            else:
                response_type_variable_list = FIELDS_BY_REPONSE_TYPE[
                    request.form[study_level_key]]
        else:
            study_level_var_list.append(
                (study_level_key, request.form[study_level_key]))
    LOGGER.debug(f'THIS IS THE LIST: {study_level_var_list}')
    covariates = []
    # search for the covariates
    for key in request.form:
        if key.startswith('covariate_name_'):
            # Extract the index
            index = key.split('_')[-1]
            category_key = f'covariate_category_{index}'
            covariate_name = request.form[key]
            covariate_category = request.form.get(category_key, '')
            covariates.append((covariate_name, covariate_category))

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user_ip = request.remote_addr
    additional_fields = []
    if precision_level != COORDINATE_PRECISION_FULL_PRECISION_VALUE:
        additional_fields.append('SiteID')
    variables = {
        'header': f'{datetime_str},{user_ip}',
        'study_level_variables': study_level_var_list,
        'headers': (
            additional_fields + BASE_FIELDS +
            response_type_variable_list + [
                f'Covariate_{name}' for name, _ in covariates]),
    }

    # Render the template with variables
    with open('templates/living_database_study.jinja', 'r') as file:
        template_content = file.read()
    living_database_template = Template(template_content)
    output = living_database_template.render(variables)
    LOGGER.debug(output)

    response = Response(output, mimetype='text/csv')
    # Specify the name of the download file
    filename = f"living_database_template_{datetime_str}.csv"
    response.headers['Content-Disposition'] = (
        f'attachment; filename={filename}')
    return response


def to_dict(obj):
    """
    Convert a SQLAlchemy model object into a dictionary.
    """
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
            if hasattr(Study, field):
                column = getattr(Study, field)
            elif hasattr(Sample, field):
                column = getattr(Sample, field)
            else:
                raise AttributeError(f"Field '{field}' not found in Study or Sample.")
            if operation == '=' and '*' in value:
                value = value.replace('*', '%')
            filter_condition = OPERATION_MAP[operation](column, value)
            LOGGER.debug(filter_condition)
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

        if ul_lat is not None:
            filters.append(Sample.latitude <= ul_lat)
            filters.append(Sample.latitude >= lr_lat)
            filters.append(Sample.longitude <= ul_lng)
            filters.append(Sample.longitude >= lr_lng)

        year_range = request.form.get('yearRange')
        if year_range is not None:
            year_range = year_range.strip()
            LOGGER.debug(year_range)
            if year_range != '':
                m = [
                    v.group() for v in
                    re.finditer(r"(\d+ ?- ?\d+)|(?:(\d+))", year_range)]
                year_filters = []
                for year_group in m:
                    if '-' not in year_group:
                        year = int(year_group.strip())
                        LOGGER.debug(f'{year}')
                        year_filters.append(Sample.year == year)
                    else:
                        start_year, end_year = [
                            int(v.strip()) for v in year_group.split('-')]
                        LOGGER.debug(f'{start_year} {end_year}')
                        year_filters.append(
                            and_(
                                Sample.year >= start_year,
                                Sample.year <= end_year))
                filters.append(or_(*year_filters))

        study_query = session.query(Study).join(
            Sample, Sample.study_id == Study.id_key).filter(
            and_(*filters))
        sample_query = session.query(Sample).join(
            Study, Sample.study_id == Study.id_key).filter(
            and_(*filters))

        LOGGER.debug(f'processing the result of {study_query.count()} results')
        study_query_result = [to_dict(s) for s in study_query.all()]
        sample_query_result = [to_dict(s) for s in sample_query.all()]
        LOGGER.debug('rendering the result')
        session.close()
        return render_template(
            'query_result.html',
            studies=study_query_result,
            samples=sample_query_result)
    except Exception as e:
        LOGGER.exception(f'error with {e}')
        raise


@app.route('/api/searchable_fields', methods=['GET'])
def searchable_fields():
    return FILTERABLE_FIELDS

if __name__ == '__main__':
    app.run(debug=True)
