"""Script to take in a CSV table and put it in the database."""
import argparse
import collections
import glob
from itertools import zip_longest

from concurrent.futures import ProcessPoolExecutor
from database import SessionLocal, init_db
from database_model_definitions import Study, DOI, Sample, Covariate, COVARIATE_ID, DOI_ID, STUDY_ID, StudyDOIAssociation
from sqlalchemy import inspect
from sqlalchemy import and_
import numpy
import pandas

TABLE_MAPPING_PATH = 'table_mapping.csv'

def fetch_or_create_study(session, study_id, doi, metadata):
    study = session.query(Study).join(StudyDOIAssociation).join(DOI).filter(
        and_(
            DOI.id_key == doi.id_key,
            Study.study_id == study_id
        )
    ).first()
    if study is None:
        study = Study(
            study_id=study_id,
            paper_dois=[doi],
            study_metadata=metadata)
        session.add(study)
    else:
        print(study)

    return study
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
    parser.add_argument('metadata_table_path', help='Path to metadata table for studies in samples')
    parser.add_argument('sample_table_path', help='Path to table of samples on each row')
    args = parser.parse_args()

    metadata_table = pandas.read_csv(args.metadata_table_path, low_memory=False)
    metadata_table.columns = map(str.lower, metadata_table.columns)
    metadata_table = metadata_table.dropna(subset=['study_id', 'doi'])
    def combine_columns(row):
        combined = []
        for col in row.index:
            if col not in ['Study_ID', 'DOI']:
                combined.append(f"{col}: {row[col]}")
        return "\n\n".join(combined)
    metadata_table['_combined'] = metadata_table.apply(combine_columns, axis=1)

    study_id_to_study_map = dict()
    for index, row in metadata_table.iterrows():
        doi = fetch_or_create_doi(session, row[DOI_ID])
        study = fetch_or_create_study(
            session, row[STUDY_ID], doi, row['_combined'])
        study_id_to_study_map[row[STUDY_ID]] = study

    # loop through rows
    sample_table = pandas.read_csv(
        args.sample_table_path,
        low_memory=False,
        nrows=100)
    sample_table = sample_table.loc[:, sample_table.columns.str.strip() != '']
    sample_table.columns = map(str.lower, sample_table.columns)
    for index, row in sample_table.iterrows():
        print(f'{100*index/len(sample_table.index):.2f}%')
        # one row is a sample
        extra_columns = []
        study = study_id_to_study_map[row[STUDY_ID]]
        sample = Sample(study=study)
        sample_fields = set(dir(sample))
        covariate_list = []
        for column in sample_table.columns:
            if column == STUDY_ID:
                continue
            if column in sample_fields:
                if column == 'latitude':
                    print(column)
                setattr(sample, column, row[column])
            elif column.startswith(COVARIATE_ID):
                covariate_val = row[column]
                if isinstance(covariate_val, str):
                    if not covariate_val:
                        continue
                elif numpy.isnan(row[column]):
                    continue
                covariate_name = '_'.join(column.split('_')[1:])
                covariate = Covariate(
                    sample=sample,
                    covariate_name=covariate_name,
                    covariate_value=row[column])
                covariate_list.append(covariate)
            else:
                # Track the extra column
                extra_columns.append(column)
        if extra_columns:
            raise ValueError(f'unknown extra columns: {extra_columns}')
        sample.covariates = covariate_list
        session.add(sample)
    session.commit()
    session.close()

if __name__ == '__main__':
    main()
