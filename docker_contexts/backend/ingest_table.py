"""Script to take in a CSV table and put it in the database."""
import argparse
import geopandas
import numpy
import logging
import sys

from database import SessionLocal, init_db
from database_model_definitions import Study, DOI, Sample, Covariate, Point, COVARIATE_ID, DOI_ID, STUDY_ID, StudyDOIAssociation, LATITUDE, LONGITUDE
from sqlalchemy import and_
import pandas

TABLE_MAPPING_PATH = 'table_mapping.csv'


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


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


def main():
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='parse table')
    parser.add_argument('metadata_table_path', help='Path to metadata table for studies in samples')
    parser.add_argument('sample_table_path', help='Path to table of samples on each row')
    parser.add_argument('--nrows', type=int, default=None, help='to limit the number of rows for testing')
    args = parser.parse_args()

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
