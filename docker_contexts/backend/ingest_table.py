"""Script to take in a CSV table and put it in the database."""
import time
from difflib import get_close_matches
import io
import os
import argparse
import geopandas
import numpy
import logging
import sys

import chardet
from database import SessionLocal, init_db
from database_model_definitions import CovariateDefn
from database_model_definitions import CovariateValue
from database_model_definitions import CovariateAssociation, CovariateType
from database_model_definitions import Point
from database_model_definitions import Geolocation
from database_model_definitions import Study
from database_model_definitions import Sample
from database_model_definitions import REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import REQUIRED_STUDY_FIELDS
from database_model_definitions import STUDY_ID
from database_model_definitions import OBSERVATION
from database_model_definitions import LATITUDE
from database_model_definitions import LONGITUDE
from database_model_definitions import YEAR
from sqlalchemy import and_, func
from sqlalchemy.exc import NoResultFound
import pandas as pd


TABLE_MAPPING_PATH = 'table_mapping.csv'
TO_ADD_BUFFER = []
OBJECT_CACHE = {}
STUDY_CACHE = {}

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

#@profile
def validate_tables(study_table_df, sample_table_df):
    # all the sample study ids should be in the metadata table
    study_ids_metadata = set(study_table_df['study_id'].unique())
    study_ids_sample = set(sample_table_df['study_id'].unique())

    missing_in_sample = study_ids_sample-study_ids_metadata
    if missing_in_sample:
        LOGGER.warning(
            f'these study ids are in sample but not in metadata: '
            f'{missing_in_sample}')

    return True


def read_csv_with_detected_encoding(file_path, **kwargs):
    # Detect the encoding
    if not isinstance(file_path, str):
        return read_csv_with_detected_encoding_from_stringio(file_path, **kwargs)
    with open(file_path, 'rb') as f:
        raw_data = f.read(300000)  # Read the first 100,000 bytes
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    if not encoding:
        raise ValueError("Failed to detect encoding")

    # Read the CSV file with the detected encoding
    encodings_to_try = ['utf-8']
    while True:
        try:
            LOGGER.info(f'attempting to read with encoding {file_path} {encoding}')
            df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace', **kwargs)
            LOGGER.info(f"Successfully read {file_path} with detected encoding: {encoding}")
            return df
        except UnicodeDecodeError as e:
            if encodings_to_try:
                encoding = encodings_to_try.pop()
            raise ValueError(f"Failed to read the file with detected encoding: {encoding}") from e


def read_csv_with_detected_encoding_from_stringio(stringio_obj, nrows=None):
    # Convert StringIO to BytesIO
    stringio_obj.seek(0)
    raw_data = stringio_obj.read().encode()  # Encode the string to bytes
    bytesio_obj = io.BytesIO(raw_data)

    # Detect the encoding
    bytesio_obj.seek(0)
    sample = bytesio_obj.read(10000)  # Read the first 10,000 bytes
    result = chardet.detect(sample)
    encoding = result['encoding']

    if not encoding:
        raise ValueError("Failed to detect encoding")

    # Use the detected encoding to read the CSV data
    stringio_obj.seek(0)
    df = pd.read_csv(stringio_obj, encoding=encoding, nrows=nrows)
    print(f"Successfully read the file with detected encoding: {encoding}")
    return df


#@profile
def fetch_or_create_study(session, covariate_defn_list, row):
    global OBJECT_CACHE
    global STUDY_CACHE
    global TO_ADD_BUFFER
    study_id = row[STUDY_ID]
    study = None
    if study_id in STUDY_CACHE:
        study = STUDY_CACHE[study_id]
    else:
        study = session.query(Study).filter(
            Study.name == study_id).first()
        if study is not None:
            STUDY_CACHE[study_id] = study
            return study

    study = Study(name=study_id)
    session.add(study)
    STUDY_CACHE[study_id] = study
    for covariate_defn in covariate_defn_list:
        covariate_name = covariate_defn.name
        if covariate_defn.name not in row:
            continue

        covariate_value = str(row[covariate_name])

        covariate_id_tuple = (covariate_name, covariate_value)
        if covariate_id_tuple in OBJECT_CACHE:
            covariate_value_obj = OBJECT_CACHE[covariate_id_tuple]
        else:
            covariate_value_obj = (
                session.query(CovariateValue)
                .join(CovariateDefn)
                .filter(CovariateDefn.name == covariate_name,
                        CovariateValue.value == covariate_value).first())
            if covariate_value_obj is not None:
                OBJECT_CACHE[covariate_id_tuple] = covariate_value_obj
            else:
                covariate_value_obj = CovariateValue(
                    value=covariate_value,
                    covariate_defn=covariate_defn)
                OBJECT_CACHE[covariate_id_tuple] = covariate_value_obj
                session.add(covariate_value_obj)
        if covariate_value_obj not in study.covariates:
            study.covariates.append(covariate_value_obj)
    return study


#@profile
def define_new_covariates(session, table_source_path, covariate_names, covariate_association):
    for name in covariate_names:
        existing = session.query(CovariateDefn).filter_by(
            name=name).first()
        if existing:
            continue
        hidden_covariate = CovariateDefn(
            display_order=999,
            name=name,
            editable_name=True,
            covariate_type=CovariateType.STRING.value,
            covariate_association=covariate_association,
            description=f'uncorrelated with existing covariates during ingestion. found in {table_source_path}',
            queryable=True,
            always_display=False,
            condition=None,
            hidden=False,
            show_in_point_table=False,
            search_by_unique=False,
        )
        session.add(hidden_covariate)
    session.commit()
    session.flush()

#@profile
def fetch_or_create_geolocation_name(session, name, geolocation_type):
    global OBJECT_CACHE
    global TO_ADD_BUFFER
    geolocation_id_tuple = (name, geolocation_type)
    if geolocation_id_tuple in OBJECT_CACHE:
        return OBJECT_CACHE[geolocation_id_tuple]
    geolocation = (
        session.query(Geolocation)
        .filter(Geolocation.geolocation_name == name,
                Geolocation.geolocation_type == geolocation_type)).first()
    if geolocation is not None:
        OBJECT_CACHE[geolocation_id_tuple] = geolocation
        return geolocation
    geolocation = Geolocation(
        geolocation_name=name,
        geolocation_type=geolocation_type)
    OBJECT_CACHE[geolocation_id_tuple] = geolocation
    TO_ADD_BUFFER.append(geolocation)
    return geolocation


#@profile
def fetch_or_add_point(
        session, continent_vector, country_vector, latitude, longitude):
    global OBJECT_CACHE
    global TO_ADD_BUFFER
    lat_lng_tuple = (latitude, longitude)
    if lat_lng_tuple in OBJECT_CACHE:
        return OBJECT_CACHE[lat_lng_tuple]
    point = (
        session.query(Point)
        .filter(Point.latitude == latitude,
                Point.longitude == longitude).first())
    if point:
        OBJECT_CACHE[lat_lng_tuple] = point
        return point

    latlng_df = pd.DataFrame(
        {'longitude': [longitude], 'latitude': [latitude]})
    point_gdf = geopandas.GeoDataFrame(
        latlng_df, geometry=geopandas.points_from_xy(
            latlng_df.longitude, latlng_df.latitude), crs='EPSG:4326')
    projected_point_gdf = point_gdf.to_crs(continent_vector.crs)

    geolocation_list = []
    try:
        continent_name = None
        continent_result = geopandas.sjoin_nearest(
            projected_point_gdf, continent_vector, how='left')
        continent_name = continent_vector.at[
            continent_result.index_right.dropna().astype(int).values[0],
            'CONTINENT']
        continent_geolocation = fetch_or_create_geolocation_name(
            session, continent_name, 'CONTINENT')
        geolocation_list.append(continent_geolocation)
    except IndexError:
        raise ValueError(f'could not find a continent for {point_gdf}')

    projected_point_gdf = point_gdf.to_crs(country_vector.crs)
    try:
        country_name = None
        country_result = geopandas.sjoin_nearest(
            projected_point_gdf, country_vector, how='left')
        country_name = country_vector.at[
            country_result.index_right.dropna().astype(int).values[0],
            'nev_name']
        country_geolocation = fetch_or_create_geolocation_name(
            session, country_name, 'COUNTRY')
        geolocation_list.append(country_geolocation)
    except IndexError:
        raise ValueError(f'could not find a country for {projected_point_gdf}')

    new_point = Point(
        latitude=latitude,
        longitude=longitude,
        geolocations=geolocation_list,
        samples=[],
    )
    OBJECT_CACHE[lat_lng_tuple] = new_point
    TO_ADD_BUFFER.append(new_point)
    return new_point

#@profile
def load_column_names(table_path):
    df = read_csv_with_detected_encoding(table_path, nrows=0)
    return df.columns.tolist()

#@profile
def rootbasename(path):
    return os.path.splitext(os.path.basename(path))[0]

#@profile
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
    return sorted(
        match_result,
        key=lambda item: (
            not item[0].startswith('*'),
            item[0] == '',
            item[0].lower()))

#@profile
def extract_column_matching(matching_df, column_matching_path):
    missing_fields = []
    unmatched_field = []
    base_to_target_map = {}
    for row in matching_df.iterrows():
        expected_field, user_field = row[1]
        if isinstance(expected_field, str):
            required = expected_field.startswith('*')
            expected_field = expected_field.replace('*', '')
            if required and not isinstance(user_field, str):
                missing_fields.append(expected_field)
            base_to_target_map[user_field] = expected_field
        elif isinstance(user_field, str):
            unmatched_field.append(user_field)

    if missing_fields:
        raise ValueError(
            'was expecting the following fields in study table but they were '
            f'undefined: {missing_fields}. Fix by modifying '
            f'`{column_matching_path}` so these fields are defined.')
    return base_to_target_map, unmatched_field

#@profile
def create_matching_table(session, args, column_matching_path):
    study_covariate_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.STUDY.value).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()
    study_covariate_list = [item for item in study_covariate_list if item.name not in REQUIRED_STUDY_FIELDS]
    LOGGER.debug(study_covariate_list)
    LOGGER.debug(REQUIRED_STUDY_FIELDS)

    study_columns = load_column_names(args.study_table_path)
    matches = match_strings(
        study_columns,
        [f'*{c}' for c in REQUIRED_STUDY_FIELDS] +
        [c.name for c in study_covariate_list])
    study_matching_df = pd.DataFrame(
        matches, columns=['EXPECTED (do not change), "*" means required', 'USER INPUT (match with EXPECTED)'])

    sample_covariate_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.SAMPLE.value).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()
    sample_covariate_list = [item for item in sample_covariate_list if item.name not in REQUIRED_SAMPLE_INPUT_FIELDS]
    LOGGER.debug(sample_covariate_list)
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


TO_ADD_BUFFER = []

#@profile
def main():
    global OBJECT_CACHE
    global TO_ADD_BUFFER
    global STUDY_CACHE
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='parse table')
    parser.add_argument('study_table_path', help='Path to study table for studies in samples')
    parser.add_argument('sample_table_path', help='Path to table of samples on each row')
    parser.add_argument('--n_dataset_rows', type=int, nargs='+', default=None, help='to limit the number of rows for testing')

    args = parser.parse_args()

    if args.n_dataset_rows is None:
        skiprows = None
        nrows = None
    elif len(args.n_dataset_rows) == 1:
        skiprows = None
        nrows = args.n_dataset_rows[0]
    elif len(args.n_dataset_rows) == 2:
        skiprows = lambda x: x > 0 and x < args.n_dataset_rows[0]
        nrows = args.n_dataset_rows[1]-args.n_dataset_rows[0]

    column_matching_path = (
        f'{rootbasename(args.study_table_path)}_'
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

    study_matching_df = read_csv_with_detected_encoding(io.StringIO(''.join(lines[1:sample_table_start_index])))
    sample_matching_df = read_csv_with_detected_encoding(io.StringIO(''.join(lines[sample_table_start_index+1:])))

    study_base_to_user_fields, raw_study_user_fields = extract_column_matching(
        study_matching_df, column_matching_path)

    # read metadata table and rename the columns to expected names, drop
    # NAs
    study_table_df = read_csv_with_detected_encoding(args.study_table_path, low_memory=False)
    study_table_df.rename(columns=study_base_to_user_fields, inplace=True)

    sample_base_to_user_fields, raw_sample_user_fields = extract_column_matching(
        sample_matching_df, column_matching_path)
    sample_table_df = read_csv_with_detected_encoding(
        args.sample_table_path, low_memory=False, skiprows=skiprows, nrows=nrows)

    sample_table_df.rename(columns=sample_base_to_user_fields, inplace=True)

    columns_to_cast =  [LATITUDE, LONGITUDE, OBSERVATION, YEAR]
    for column in columns_to_cast:
        sample_table_df[column] = pd.to_numeric(sample_table_df[column], errors='coerce')
    sample_table_df.dropna(subset=columns_to_cast, inplace=True)

    # drop the columns that aren't being used

    define_new_covariates(session, args.study_table_path, raw_study_user_fields, CovariateAssociation.STUDY.value)
    define_new_covariates(session, args.sample_table_path, raw_sample_user_fields, CovariateAssociation.SAMPLE.value)
    validate_tables(study_table_df, sample_table_df)

    country_vector = geopandas.read_file(
        "./base_data/countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg")
    continent_vector = geopandas.read_file('./base_data/continents.gpkg')
    if continent_vector.crs.is_geographic:
        # Reproject continent_vector to a suitable projected CRS (e.g., EPSG:3857 for Web Mercator)
        projected_crs = 'EPSG:3857'
        continent_vector = continent_vector.to_crs(projected_crs)
    if country_vector.crs.is_geographic:
        # Reproject country_vector to a suitable projected CRS (e.g., EPSG:3857 for Web Mercator)
        projected_crs = 'EPSG:3857'
        country_vector = country_vector.to_crs(projected_crs)

    session.commit()
    study_covariate_defn_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.STUDY.value).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()

    study_id_to_study_map = {}
    for index, row in study_table_df.iterrows():
        study_id = row[STUDY_ID]
        if not isinstance(study_id, str) and (study_id is None or numpy.isnan(study_id)):
            continue
        study_id_to_study_map[row[STUDY_ID]] = fetch_or_create_study(
            session, study_covariate_defn_list, row)

    sample_covariate_defn_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.SAMPLE.value).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()
    start_time = time.time()
    for index, row in sample_table_df.iterrows():
        if index % 1000 == 0 and index > 1:
            time_so_far = time.time() - start_time
            time_per_element = time_so_far/(index+1)
            time_left = (len(sample_table_df.index)-index)*time_per_element
            print(
                f'{100*index/len(sample_table_df.index):.2f}% '
                f'{index}/{len(sample_table_df.index)} time left: {time_left:.2f}s')
            session.add_all(TO_ADD_BUFFER)
            session.flush()
            session.commit()
            OBJECT_CACHE = {}
            TO_ADD_BUFFER[:] = []
        if any([isinstance(row[x], str) or numpy.isnan(row[x]) for x in [LATITUDE, LONGITUDE, OBSERVATION]]):
            LOGGER.warning(
                f'found a row at {index} with no lat/long/or observation value: '
                f'{[(x, row[x]) for x in [LATITUDE, LONGITUDE, OBSERVATION]]}, skipping')
            continue
        point = fetch_or_add_point(
            session, continent_vector, country_vector,
            row[LATITUDE],
            row[LONGITUDE])

        study_id = row[STUDY_ID]
        try:
            study = STUDY_CACHE[study_id]
        except KeyError:
            LOGGER.warning(f'error on this row: {index} {row}')
            continue

        sample = Sample(
            point=point,
            study=study,
            observation=row[OBSERVATION]
            )
        session.add(sample)
        for covariate_defn in sample_covariate_defn_list:
            covariate_name = covariate_defn.name
            if covariate_name not in row:
                continue
            covariate_value = row[covariate_name]
            if not isinstance(covariate_value, str) and numpy.isnan(
                    covariate_value):
                continue
            covariate_value_obj = CovariateValue(
                value=str(covariate_value),
                covariate_defn=covariate_defn)
            session.add(covariate_value_obj)
            sample.covariates.append(covariate_value_obj)

    # add all samples in samples_to_add
    print('bulk inserting remainder')
    session.add_all(TO_ADD_BUFFER)
    del TO_ADD_BUFFER
    print('about to commit ')
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
