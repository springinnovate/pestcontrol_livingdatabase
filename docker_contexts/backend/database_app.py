import configparser

from database import SessionLocal
from database_model_definitions import BASE_FIELDS
from database_model_definitions import Study, Sample, Covariate
from database_model_definitions import STUDY_LEVEL_VARIABLES
from flask import Flask
from flask import render_template
from flask import request
from jinja2 import Template
from sqlalchemy import inspect


config = configparser.ConfigParser()
config.read('alembic.ini')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = config['alembic']['sqlalchemy.url']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


@app.route('/')
def index():
    session = SessionLocal()
    # Example: Fetch all records from ExampleModel
    samples = session.query(Sample).all()
    return 'Number of samples: ' + str(len(samples))


@app.route('/home')
def home():
    inspector = inspect(Study)
    print(inspector.columns.id_key)
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

    return render_template(
        'query_builder.html',
        fields=study_columns+sample_columns+covariate_columns)


@app.route('/template')
def template():
    return render_template(
        'template_builder.html',
        study_level_fields=STUDY_LEVEL_VARIABLES)


@app.route('/build_template', methods=['POST'])
def build_template():
    data = {}
    # Loop through the request.form dictionary
    study_level_var_list = []
    for study_level_key in STUDY_LEVEL_VARIABLES:
        study_level_var_list.append(
            (study_level_key, request.form[study_level_key]))

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

    variables = {
        'study_level_variables': study_level_var_list,
        'headers': BASE_FIELDS + [
            f'Covariate_{name}' for name, _ in covariates],
    }

    # Render the template with variables
    with open('templates/living_database_study.jinja', 'r') as file:
        template_content = file.read()
    living_database_template = Template(template_content)
    output = living_database_template.render(variables)
    print(output)
    return output


@app.route('/process_query', methods=['POST'])
def process_query():
    fields = request.form.getlist('field')
    operations = request.form.getlist('operation')
    values = request.form.getlist('value')

    session = SessionLocal()
    # Example: Fetch all records from ExampleModel
    samples = session.query(Sample).all()

    # Example of how you might process these queries
    for field, operation, value in zip(fields, operations, values):
        print(f"query: {field} {operation} {value}")
        # Build your query based on the input
    return "Query processed"


if __name__ == '__main__':
    app.run(debug=True)
