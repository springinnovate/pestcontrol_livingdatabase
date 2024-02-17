from flask import Flask
from flask import render_template
from flask import request

import configparser
from database import SessionLocal, init_db
from database_model_definitions import Study, DOI, Sample, Covariate, COVARIATE_ID, DOI_ID
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
