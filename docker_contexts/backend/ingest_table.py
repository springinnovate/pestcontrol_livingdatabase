"""Script to take in a CSV table and put it in the database."""
import argparse
import collections
import glob
from itertools import zip_longest

from concurrent.futures import ProcessPoolExecutor
from database import SessionLocal, init_db
from database_model_definitions import Study, DOI, Sample, Covariate, COVARIATE_ID, DOI_ID
from sqlalchemy import inspect
import numpy
import pandas

TABLE_MAPPING_PATH = 'table_mapping.csv'


def fetch_or_create_doi(session, doi_value):
    # Attempt to find an existing DOI with the given value
    existing_doi = session.query(DOI).filter(DOI.doi == doi_value).first()

    if existing_doi is None:
        # DOI doesn't exist, so create a new one
        new_doi = DOI(doi=doi_value)
        session.add(new_doi)
        print("Created new DOI:", new_doi.doi)
    else:
        # Use the existing DOI
        new_doi = existing_doi
        print("Using existing DOI:", new_doi.doi)
    return new_doi

def main():
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='parse table')
    parser.add_argument('table_path', help='Path to table of samples')
    args = parser.parse_args()

    table = pandas.read_csv(args.table_path)

    print(table.columns)
    return
    inspector = inspect(Study)
    print(inspector.columns.id_key)
    study_columns = [
        (column.name, column.nullable) for column in inspector.columns
        if not column.primary_key]
    inspector = inspect(Sample)
    sample_columns = [
        (column.name, column.nullable) for column in inspector.columns
        if not column.primary_key]
    inspector = inspect(Covariate)
    covariate_columns = [
        (column.name, column.nullable) for column in inspector.columns
        if not column.primary_key]

    table_to_database_column_map = {}
    database_to_table_map = {}
    for index, row in column_mapping_table.iterrows():
        table_to_database_column_map[row['input columns']] = \
            row['database column']
        database_to_table_map[row['database column']] = \
            row['input columns']

    study_map = {}
    sample_by_study_list = collections.defaultdict(list)
    for index, row in sample_table.iterrows():
        if index > 1000:
            break
        print(f'{100*index/len(sample_table.index):.2f}%')
        new_doi = None
        sample_dict = dict()
        covariate_dict_list = []
        for input_col, db_col in table_to_database_column_map.items():
            if db_col == DOI_ID:
                new_doi = fetch_or_create_doi(session, row[input_col])
                if new_doi is not None:
                    sample_dict[input_col] = new_doi
                continue
            if db_col == COVARIATE_ID:
                val = row[input_col]
                if isinstance(val, float) and numpy.isnan(val):
                    val = 'NA'
                new_covariate = {
                    'covariate_category': None,
                    'covariate_name': input_col,
                    'covariate_value': val}
                covariate_dict_list.append(new_covariate)
                continue
            sample_dict[db_col] = row[input_col]
        study_dict = {}

        for study_field in study_columns:
            value = None
            if (study_field[0] in database_to_table_map and
                    database_to_table_map[study_field[0]] in row):
                value = row[database_to_table_map[study_field[0]]]
            study_dict[study_field[0]] = value

        study_tuple = tuple(study_dict)
        if study_tuple not in study_map:
            study_map[study_tuple] = Study(**study_dict)
        sample = Sample(**sample_dict)

        covariate_list = []
        for covariate_dict in covariate_dict_list:
            covariate = Covariate(**covariate_dict)
            covariate_list.append(covariate)
        sample.covariates = covariate_list
        sample_by_study_list[study_tuple].append(sample)
        session.add(sample)

    for study_tuple in sample_by_study_list:
        study_map[study_tuple].samples = sample_by_study_list[study_tuple]
        session.add(study_map[study_tuple])

    session.commit()
    return
    print(df.columns)

    with open('table_mapping.csv', 'w') as table:
        table.write('input columns,database column,base columns,required\n')
        for column_val, csv_column_name in zip_longest(
                study_columns+sample_columns+covariate_columns,
                df.columns
                ):
            if column_val is None:
                column_name, optional_column = '', ''
            else:
                column_name, optional_column = column_val
            if column_name == 'id_key':
                continue
            table.write(f'{csv_column_name},,{column_name},{optional_column}\n')

    return
    with ProcessPoolExecutor() as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_pdf, file_path)
            future_list.append(future)
        article_list = [
            article for future in future_list for article in future.result()]
    upsert_articles(db, article_list)
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
