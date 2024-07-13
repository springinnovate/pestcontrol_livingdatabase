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
from database_model_definitions import CovariateDefn
from database_model_definitions import CovariateValue
from database_model_definitions import CovariateAssociation
from database_model_definitions import Point
from database_model_definitions import GeolocationName
from database_model_definitions import Study
from database_model_definitions import Sample
from database_model_definitions import REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import REQUIRED_STUDY_FIELDS
from database_model_definitions import STUDY_ID
from database_model_definitions import OBSERVATION
from database_model_definitions import LATITUDE
from database_model_definitions import LONGITUDE
from sqlalchemy import and_, func
from sqlalchemy.exc import NoResultFound
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


def validate_tables(study_table_df, sample_table_df):
    error_str = ''
    # all the sample study ids should be in the metadata table
    study_ids_metadata = set(study_table_df['study_id'].unique())
    study_ids_sample = set(sample_table_df['study_id'].unique())

    missing_in_sample = study_ids_sample-study_ids_metadata
    if missing_in_sample:
        error_str += (
            f'these study ids are in metadata but not in sample: '
            f'{missing_in_sample}')

    if error_str:
        raise ValueError(error_str)
    return True


def fetch_or_create_study(session, covariate_defn_list, row):
    study_id = row[STUDY_ID]
    study = session.query(Study).filter(Study.id_key == study_id).first()
    if study is not None:
        return study

    study = Study(id_key=study_id)
    for covariate_defn in covariate_defn_list:
        if covariate_defn.name not in row:
            # it's possible to have covariates that aren't in the set
            continue
        value = row[covariate_defn.name]

        covariate_value = session.query(CovariateValue).filter(
            CovariateValue.covariate_defn == covariate_defn,
            CovariateValue.value == value).first()
        if covariate_value is None:
            covariate_value = CovariateValue(
                value=value,
                covariate_defn=covariate_defn,
                studies=[study])
            session.add(covariate_value)
        elif study not in covariate_value.studies:
            covariate_value.studies.append(study)

    session.add(study)
    return study


def fetch_or_create_geolocation_name(session, name):
    try:
        geolocation_name = session.query(GeolocationName).filter_by(
            geolocation_name=name).one()
    except NoResultFound:
        geolocation_name = GeolocationName(geolocation_name=name)
        session.add(geolocation_name)
    return geolocation_name


def fetch_or_add_point(
        session, continent_vector, country_vector, latitude, longitude):
    existing_point = session.query(Point).filter_by(
        latitude=latitude, longitude=longitude).first()

    if existing_point:
        return existing_point

    latlng_df = pd.DataFrame(
        {'longitude': [longitude], 'latitude': [latitude]})
    point_gdf = geopandas.GeoDataFrame(
        latlng_df, geometry=geopandas.points_from_xy(
            latlng_df.longitude, latlng_df.latitude), crs='EPSG:4326')
    if continent_vector.crs.is_geographic:
        # Reproject continent_vector to a suitable projected CRS (e.g., EPSG:3857 for Web Mercator)
        projected_crs = 'EPSG:3857'
        continent_vector = continent_vector.to_crs(projected_crs)
        projected_point_gdf = point_gdf.to_crs(projected_crs)
    else:
        projected_point_gdf = point_gdf

    geolocation_list = []
    try:
        continent_name = None
        continent_result = geopandas.sjoin_nearest(
            projected_point_gdf, continent_vector, how='left')
        continent_name = continent_vector.at[
            continent_result.index_right.dropna().astype(int).values[0],
            'CONTINENT']
        continent_geolocation = fetch_or_create_geolocation_name(
            session, continent_name)
        geolocation_list.append(continent_geolocation)
    except IndexError:
        raise ValueError(f'could not find a continent for {point_gdf}')

    if country_vector.crs.is_geographic:
        # Reproject country_vector to a suitable projected CRS (e.g., EPSG:3857 for Web Mercator)
        projected_crs = 'EPSG:3857'
        country_vector = country_vector.to_crs(projected_crs)
        projected_point_gdf = point_gdf.to_crs(projected_crs)
    else:
        projected_point_gdf = point_gdf
    try:
        country_name = None
        country_result = geopandas.sjoin_nearest(
            projected_point_gdf, country_vector, how='left')
        country_name = country_vector.at[
            country_result.index_right.dropna().astype(int).values[0],
            'nev_name']
        country_geolocation = fetch_or_create_geolocation_name(
            session, country_name)
        geolocation_list.append(country_geolocation)
    except IndexError:
        raise ValueError(f'could not find a country for {projected_point_gdf}')

    new_point = Point(
        latitude=latitude,
        longitude=longitude,
        geolocations=geolocation_list
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
        expected_field, user_field = row[1]
        if any([not isinstance(field_set, str) and (field_set is None or numpy.isnan(field_set))
                for field_set in [expected_field, user_field]]):
            continue
        required = expected_field.startswith('*')
        expected_field = expected_field.replace('*', '')

        if required and not isinstance(user_field, str):
            missing_fields.append(expected_field)
        else:
            base_to_target_map[user_field] = expected_field
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

    study_columns = load_column_names(args.study_table_path)
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
    parser.add_argument('study_table_path', help='Path to study table for studies in samples')
    parser.add_argument('sample_table_path', help='Path to table of samples on each row')
    parser.add_argument('--column_matching_table_path', help='Path to column matching table.')
    parser.add_argument('--nrows', type=int, default=None, help='to limit the number of rows for testing')

    args = parser.parse_args()

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

    study_matching_df = pd.read_csv(io.StringIO(''.join(lines[1:sample_table_start_index])))
    sample_matching_df = pd.read_csv(io.StringIO(''.join(lines[sample_table_start_index+1:])))

    study_base_to_user_fields = extract_column_matching(
        study_matching_df, column_matching_path)

    # read metadata table and rename the columns to expected names, drop
    # NAs
    study_table_df = pd.read_csv(args.study_table_path, low_memory=False)
    study_table_df.rename(columns=study_base_to_user_fields, inplace=True)
    study_table_df = study_table_df[study_base_to_user_fields.values()].dropna()

    sample_base_to_user_fields = extract_column_matching(
        sample_matching_df, column_matching_path)
    sample_table_df = pd.read_csv(args.sample_table_path, low_memory=False, nrows=args.nrows)
    sample_table_df.rename(columns=sample_base_to_user_fields, inplace=True)
    # drop the columns that aren't being used
    sample_table_df = sample_table_df[sample_base_to_user_fields.values()]

    validate_tables(study_table_df, sample_table_df)

    country_vector = geopandas.read_file(
        "../../base_data/countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg")
    continent_vector = geopandas.read_file('../../base_data/continents.gpkg')

    study_covariate_defn_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.STUDY).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()

    study_id_to_study_map = {}
    for index, row in study_table_df.iterrows():
        study_id = row[STUDY_ID]
        study_id_to_study_map[row[STUDY_ID]] = fetch_or_create_study(
            session, study_covariate_defn_list, row)
    session.flush()

    sample_covariate_defn_list = session.query(CovariateDefn).filter(
        CovariateDefn.covariate_association == CovariateAssociation.SAMPLE).order_by(
        CovariateDefn.display_order,
        func.lower(CovariateDefn.name)).all()
    for index, row in sample_table_df.iterrows():
        print(f'{100*index/len(sample_table_df.index):.2f}%')
        if any([numpy.isnan(row[x]) for x in [LATITUDE, LONGITUDE]]):
            raise ValueError(
                f'found a row at {index} with no lat or long coordinates: '
                f'{row}')
            continue
        point = fetch_or_add_point(
            session, continent_vector, country_vector,
            row[LATITUDE],
            row[LONGITUDE])

        study_id = row[STUDY_ID]
        study = study_id_to_study_map[study_id]
        sample = Sample(
            point=point,
            study=study,
            observation=row[OBSERVATION]
            )
        session.add(sample)

        for covariate_defn in sample_covariate_defn_list:
            if covariate_defn.name not in row:
                # it's possible to have covariates that aren't in the set
                continue
            value = row[covariate_defn.name]
            if not isinstance(value, str) and numpy.isnan(value):
                continue

            covariate_value = session.query(CovariateValue).filter(
                CovariateValue.covariate_defn == covariate_defn,
                CovariateValue.value == value).first()
            if covariate_value is None:
                covariate_value = CovariateValue(
                    value=value,
                    covariate_defn=covariate_defn,
                    samples=[sample])
                session.add(covariate_value)
            elif sample not in covariate_value.samples:
                covariate_value.samples.append(sample)
        session.add(sample)
        session.flush()

    print('about to commit ')
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
