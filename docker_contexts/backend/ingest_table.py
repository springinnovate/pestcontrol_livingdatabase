"""Script to take in a CSV table and put it in the database."""
from difflib import get_close_matches
import io
import os
import argparse
import geopandas
import numpy
import logging
import sys

from database import SessionLocal, init_db
from database_model_definitions import CovariateDefn, CovariateAssociation
from database_model_definitions import REQUIRED_SAMPLE_INPUT_FIELDS, REQUIRED_STUDY_FIELDS
from sqlalchemy import or_, and_, func
import pandas as pd


TABLE_MAPPING_PATH = 'table_mapping.csv'


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def validate_tables(metadata_table_df, sample_table_df):
    error_str = ''
    # should have same study ids
    study_ids_metadata = set(metadata_table_df['study_id'].unique())
    study_ids_sample = set(sample_table_df['study_id'].unique())

    missing_in_sample = study_ids_metadata-study_ids_sample
    if missing_in_sample:
        error_str += (
            f'these study ids are in metadata but not in sample: '
            f'{missing_in_sample}')

    missing_in_metadata = study_ids_sample-study_ids_metadata
    if missing_in_metadata:
        error_str += (
            f'these study ids are in study but not in metadata: '
            f'{missing_in_metadata}')

    if error_str:
        raise ValueError(error_str)
    return True


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


def fetch_or_add_point(
        session, continent_vector, country_vector, latitude, longitude):
    # Check if the point already exists
    existing_point = session.query(Point).filter_by(
        latitude=latitude, longitude=longitude).first()

    if existing_point:
        return existing_point

    latlng_df = pandas.DataFrame(
        {'longitude': [longitude], 'latitude': [latitude]})
    point_gdf = geopandas.GeoDataFrame(
        latlng_df, geometry=geopandas.points_from_xy(
            latlng_df.longitude, latlng_df.latitude), crs=continent_vector.crs)
    try:
        continent_name = None
        continent_result = geopandas.sjoin(
            point_gdf, continent_vector, how='left', predicate='within')
        if not continent_result.empty:
            continent_name = continent_vector.at[
                continent_result.index_right.dropna().astype(int).values[0],
                'CONTINENT']
    except IndexError:
        continent_name = 'ocean'

    try:
        country_name = None
        country_result = geopandas.sjoin(
            point_gdf, country_vector, how='left', predicate='within')
        if not country_result.empty:
            country_name = country_vector.at[
                country_result.index_right.dropna().astype(int).values[0],
                'nev_name']
    except IndexError:
        country_name = 'ocean'

    new_point = Point(
        latitude=latitude,
        longitude=longitude,
        country=country_name,
        continent=continent_name
    )

    session.add(new_point)
    return new_point


def load_column_names(table_path):
    df = pd.read_csv(table_path, nrows=0)
    return df.columns.tolist()


def rootbasename(path):
    return os.path.splitext(os.path.basename(path))[0]


def match_strings(base_list, to_match_list):
    match_result = []
    matched_so_far = set()
    remaining_matches = set(to_match_list)
    for base_str in base_list:
        match = get_close_matches(
            base_str.replace('*', ''), remaining_matches, n=1, cutoff=0.6)
        if match:
            match = match[0]
            if match in matched_so_far:
                continue
            match_result.append((match, base_str))
            remaining_matches.remove(match)
        else:
            match_result.append(('', base_str))
    for remaining_str in remaining_matches:
        match_result.append((remaining_str, ''))
    print(match_result)
    return sorted(
        match_result,
        key=lambda item: (
            not item[0].startswith('*'),
            item[0] == '',
            item[0].lower()))


def extract_column_matching(matching_df, column_matching_path):
    missing_fields = []
    base_to_target_map = {}
    for row in matching_df.iterrows():
        expected_field, base_field = row[1]
        if expected_field in [None, numpy.nan]:
            continue
        required = expected_field.startswith('*')
        expected_field = expected_field.replace('*', '')

        if required and not isinstance(base_field, str):
            missing_fields.append(expected_field)
        else:
            base_to_target_map[base_field] = expected_field
    if missing_fields:
        raise ValueError(
            'was expecting the following fields in study table but they were '
            f'undefined: {missing_fields}. Fix by modifying '
            f'`{column_matching_path}` so these fields are defined.')
    return base_to_target_map


def create_matching_table(session, args, column_matching_path):
    study_covariate_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.STUDY).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()

    study_columns = load_column_names(args.metadata_table_path)
    matches = match_strings(
        study_columns,
        [f'*{c}' for c in REQUIRED_STUDY_FIELDS] +
        [c.name for c in study_covariate_list])
    study_matching_df = pd.DataFrame(
        matches, columns=['EXPECTED (do not change), "*" means required', 'USER INPUT (match with EXPECTED)'])

    sample_covariate_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.SAMPLE).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()

    sample_columns = load_column_names(args.sample_table_path)
    matches = match_strings(
        sample_columns,
        [f'*{c}' for c in REQUIRED_SAMPLE_INPUT_FIELDS] +
        [c.name for c in sample_covariate_list])

    sample_matching_df = pd.DataFrame(
        matches, columns=['EXPECTED (do not change), "*" means required', 'USER INPUT (match with EXPECTED)'])

    with open(column_matching_path, 'w', newline='') as f:
        f.write('STUDY TABLE\n')
        study_matching_df.to_csv(f, index=False)
        f.write('\n')  # Add a blank line between tables

        # Write the second dataframe to the same CSV file
        f.write('SAMPLE TABLE\n')
        sample_matching_df.to_csv(f, index=False)


def main():
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='parse table')
    parser.add_argument('metadata_table_path', help='Path to metadata table for studies in samples')
    parser.add_argument('sample_table_path', help='Path to table of samples on each row')
    parser.add_argument('--column_matching_table_path', help='Path to column matching table.')
    parser.add_argument('--nrows', type=int, default=None, help='to limit the number of rows for testing')

    args = parser.parse_args()

    column_matching_path = (
        f'{rootbasename(args.metadata_table_path)}_'
        f'{rootbasename(args.sample_table_path)}.csv')

    # if no column matching table
    if not os.path.exists(column_matching_path):
        create_matching_table(session, args, column_matching_path)
        return

    with open(column_matching_path, 'r') as f:
        lines = f.readlines()

    sample_table_start_index = next(
        (i for i, s in enumerate(lines)
         if s.startswith('SAMPLE TABLE')), -1)

    study_matching_df = pd.read_csv(io.StringIO(''.join(lines[1:sample_table_start_index])))
    sample_matching_df = pd.read_csv(io.StringIO(''.join(lines[sample_table_start_index+1:])))

    study_base_to_user_fields = extract_column_matching(
        study_matching_df, column_matching_path)
    print(study_base_to_user_fields)

    # read metadata table and rename the columns to expected names, drop
    # NAs
    metadata_table_df = pd.read_csv(args.metadata_table_path, low_memory=False)
    metadata_table_df.rename(columns=study_base_to_user_fields, inplace=True)
    metadata_table_df = metadata_table_df[study_base_to_user_fields.values()].dropna()
    print(metadata_table_df)

    sample_base_to_user_fields = extract_column_matching(
        sample_matching_df, column_matching_path)
    print(sample_base_to_user_fields)
    sample_table_df = pd.read_csv(args.sample_table_path, low_memory=False, nrows=1000)
    sample_table_df.rename(columns=sample_base_to_user_fields, inplace=True)
    sample_table_df = sample_table_df[sample_base_to_user_fields.values()].dropna()
    print(sample_table_df)

    # todo process the table:
    # * verify that the study ids in the study table are exclusively in the sample table
    #   error if not
    validate_tables(metadata_table_df, sample_table_df)

    return
    # load study table
    # validate that columns in study table

    # endif

    country_vector = geopandas.read_file(
        "../../base_data/countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg")
    continent_vector = geopandas.read_file('../../base_data/continents.gpkg')
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
        low_memory=False, nrows=args.nrows)
    sample_table = sample_table.loc[:, sample_table.columns.str.strip() != '']
    sample_table.columns = map(str.lower, sample_table.columns)

    covariate_data = []

    for index, row in sample_table.iterrows():
        print(f'{100*index/len(sample_table.index):.2f}%')
        study_id = row[STUDY_ID]
        if study_id not in study_id_to_study_map:
            continue
        if any([numpy.isnan(row[x]) for x in [LATITUDE, LONGITUDE]]):
            LOGGER.error(f'found a row with no lat or long coordinates: {row}')
            continue
        point = fetch_or_add_point(
            session, continent_vector, country_vector, row[LATITUDE], row[LONGITUDE])
        study = study_id_to_study_map[study_id]
        sample = Sample(study=study, point=point)
        session.add(sample)

        session.flush()
        sample_fields = set(dir(sample))
        for column, value in row.items():
            if column == STUDY_ID:
                continue
            if column in sample_fields:
                setattr(sample, column, row[column])
            elif column.startswith(COVARIATE_ID):
                if pandas.notna(value) and value != '':
                    covariate_name = '_'.join(column.split('_')[1:])
                    covariate_data.append({
                        'sample_id': sample.id_key,
                        'covariate_name': covariate_name,
                        'covariate_value': value
                    })
    print('about to do bulk insert mappings')
    session.bulk_insert_mappings(Covariate, covariate_data)
    print('about to commit ')
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
