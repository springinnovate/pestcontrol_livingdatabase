"""Sample GEE datasets given pest control CSV."""
import argparse
import bisect
import logging
import os
import collections
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import ee
import pandas

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

LIBS_TO_SILENCE = ['urllib3.connectionpool', 'googleapiclient.discovery', 'google_auth_httplib2']
for lib_name in LIBS_TO_SILENCE:
    logging.getLogger(lib_name).setLevel(logging.WARN)

POINT_ID = 'PointID'
DATASET_ID = 'MODIS/061/MCD12Q1'
VALID_YEARS = (2001, 2022)
RESOLUTION = 500
BAND_ID = 'LC_Type1'
CLASS_TABLE = {
    1: 'Evergreen Needleleaf Forests',
    2: 'Evergreen Broadleaf Forests',
    3: 'Deciduous Needleleaf Forests',
    4: 'Deciduous Broadleaf Forests',
    5: 'Mixed Forests',
    6: 'Closed Shrublands',
    7: 'Open Shrublands',
    8: 'Woody Savannas',
    9: 'Savannas',
    10: 'Grasslands',
    11: 'Permanent Wetlands',
    12: 'Croplands',
    13: 'Urban and Built-up Lands',
    14: 'Cropland/Natural Vegetation Mosaics',
    15: 'Permanent Snow and Ice',
    16: 'Barren',
    17: 'Water Bodies',
}


def get_dataset_info(dataset_row, dataset_name_field, variable_name_field):
    dataset_name = dataset_row[dataset_name_field]
    variable_name = dataset_row[variable_name_field]
    try:
        dataset = ee.ImageCollection(dataset_name).select(variable_name)
        info = dataset.first().getInfo()  # This is the blocking call we want to run in parallel.
        return (info, dataset_name, variable_name)
    except ee.ee_exception.EEException:
        dataset = ee.Image(dataset_name).select(variable_name)
        info = dataset.getInfo()  # This is the blocking call we want to run in parallel.
        return (info, dataset_name, variable_name)


def sample_dataset(year, sample_distance, point_list):
    # this works if its an image collection
    image = ee.ImageCollection(DATASET_ID).select(BAND_ID).filterDate(
        ee.Date.fromYMD(year, 1, 1)).first().rename(f'{DATASET_ID}_{year}')

    result_by_cover_type = collections.defaultdict(list)
    for i, points_chunk in enumerate(chunk_points(point_list, 5000)):
        distance_circles = ee.FeatureCollection(points_chunk).map(
            lambda feature: feature.buffer(RESOLUTION))
        landcover_masks = ee.ImageCollection(
            [image.eq(landcover_id).toFloat().rename(f'{landcover_id}')
             for landcover_id in CLASS_TABLE]).toBands()

        gaussian_kernel = ee.Kernel.gaussian(
            sample_distance, sigma=sample_distance/3, units='meters',
            normalize=True)
        landcover_average_mask = landcover_masks.convolve(gaussian_kernel)
        samples = landcover_average_mask.reduceRegions(
            collection=distance_circles,
            reducer=ee.Reducer.mean(),
            scale=RESOLUTION)
        raw_result = samples.getInfo()
        for landcover_id in CLASS_TABLE:
            # the {landcover_id-1}_{landcover_id} string comes from the fact
            # that GEE is appending a 0_ 1_ 2_ etc on the beginning of the
            # band name and the band name is the landcover code which starts
            # at 1 so 0_1 is really landcover 1
            result_by_cover_type[landcover_id].extend(
                result['properties'][f'{landcover_id-1}_{landcover_id}']
                for result in raw_result['features'])
        circles = ee.FeatureCollection(points_chunk).map(
            lambda feature: feature.buffer(RESOLUTION))
        samples = image.toFloat().reduceRegions(
                collection=circles,
                reducer=ee.Reducer.mode(),
                scale=RESOLUTION)
        raw_result = samples.getInfo()
        result_by_cover_type['lc_under_the_pixel'].extend(
            result['properties']['mode']
            for result in raw_result['features'])
    return result_by_cover_type


def chunk_points(point_list, chunk_size):
    """Yield successive chunks from point_list."""
    for i in range(0, len(point_list), chunk_size):
        yield point_list[i:i + chunk_size]


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='sample points on GEE data')
    parser.add_argument(
        'point_table_path', help='path to point sample locations')
    parser.add_argument(
        'sample_distance_m', type=int, help=(
            'Distance in meters to sample out from the point when '
            'calculating landcover proportions'))
    parser.add_argument('--authenticate', action='store_true', help='Pass this flag if you need to reauthenticate with GEE')
    parser.add_argument('--n_point_rows', type=int, help='limit csv read to this many rows')

    args = parser.parse_args()
    if args.authenticate:
        ee.Authenticate()
    ee.Initialize()

    point_table = pandas.read_csv(
        args.point_table_path,
        skip_blank_lines=True,
        converters={
            'X': lambda x: None if x == '' else float(x),
            'Y': lambda x: None if x == '' else float(x),
        },
        usecols=['X', 'Y'],
        nrows=args.n_point_rows).dropna(how='all')
    point_table[POINT_ID] = range(1, len(point_table) + 1)

    LOGGER.info(f'loaded {args.point_table_path}')

    # Create a ThreadPoolExecutor
    futures_work_list = []

    point_feature_list = [
        ee.Geometry.Point([lon, lat])
        for lon, lat in zip(
            point_table['X'],
            point_table['Y'])]

    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor.
        for year in range(VALID_YEARS[0], VALID_YEARS[1]+1):
            futures_work_list.append(
                (year, executor.submit(
                    sample_dataset,
                    year,
                    args.sample_distance_m,
                    point_feature_list)))

        # Iterate over the futures as they complete (as_completed returns them in the order they complete).
        output_table = pandas.DataFrame()
        for year, future in futures_work_list:
            LOGGER.debug(f'fetching {year} result')
            result = future.result()  # This gets the return value from get_dataset_info when it is done.
            temp_df = point_table.copy()
            temp_df['Year'] = year
            for key, val_list in result.items():
                temp_df = pandas.concat(
                    [temp_df, pandas.DataFrame({key: val_list})], axis=1)
            output_table = pandas.concat([output_table, temp_df], ignore_index=True)

    output_table = output_table.sort_values(by=[POINT_ID, 'Year'])
    output_table = output_table.drop(columns=[POINT_ID])
    input_table_basename = os.path.splitext(os.path.basename(args.point_table_path))[0]
    output_table.to_csv(
        f'{input_table_basename}_{args.sample_distance_m}.csv', index=False)

    LOGGER.info('all done')


if __name__ == '__main__':
    main()
