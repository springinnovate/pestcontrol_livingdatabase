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

PRECIP_SECTION = 'precip'


# Define a function to expand year ranges into individual years
def expand_year_range(year_field):
    def _expand_year_range(row):
        if '-' in str(row[year_field]):
            start_year, end_year = map(int, row[year_field].split('-'))
            return [{**row, year_field: str(year)} for year in range(start_year, end_year + 1)]

        else:
            return [{**row}]
    return _expand_year_range


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


def find_closest(sorted_list, num):
    # Find the point where 'num' would be inserted to keep the list sorted
    index = bisect.bisect_left(sorted_list, num)

    # If 'num' is the greatest number
    if index == len(sorted_list):
        return sorted_list[-1]
    # If 'num' is the least number
    elif index == 0:
        return sorted_list[0]
    # If 'num' is between two numbers, compare with the closest ones
    else:
        if index < len(sorted_list):
            closest_num = min(sorted_list[index - 1], sorted_list[index], key=lambda x: abs(x - num))
        else:
            closest_num = sorted_list[index - 1]
        return closest_num


def sample_dataset(dataset_name, variable_name, scale, julian_range, aggregate_function, valid_year_list, point_features_by_year):
    # Create a FeatureCollection from the list of points.

    result_samples_by_year = collections.defaultdict(list)
    # test if dataset_name is an image or image collection
    remap_value_list = None
    if 'mask' in aggregate_function:
        remap_value_list = eval(aggregate_function.split('-')[1])
        aggregate_function = 'mean'

    for year, point_list in point_features_by_year.items():
        filter_by_date = True
        if '{year}' in dataset_name:
            closest_year = find_closest(valid_year_list, year)
            local_dataset_name = dataset_name.format(year=closest_year)
            filter_by_date = False
        else:
            local_dataset_name = dataset_name

        try:
            # this works if its an image collection
            dataset = ee.ImageCollection(local_dataset_name).select(variable_name)
            _ = dataset.size().getInfo() # test if its valid
            if filter_by_date:
                julian_range_start, julian_range_end = julian_range.split('-')
                # Sample the image at each point in the FeatureCollection.
                # Create the start and end dates
                start_date = ee.Date.fromYMD(year, 1, 1).advance(
                    int(julian_range_start)-1, 'day')
                end_date = ee.Date.fromYMD(year, 1, 1).advance(
                    int(julian_range_end)-1, 'day')
                reduced_dataset = dataset.filterDate(start_date, end_date)
            else:
                reduced_dataset = dataset

            if remap_value_list is not None:
                # first reduce the by date so we don't have a bunch of images
                reduced_dataset = reduced_dataset.map(
                    lambda image: image.remap(
                        remap_value_list, [1]*len(remap_value_list)))
            image = reduced_dataset.reduce(aggregate_function).rename(
                variable_name)
        except ee.ee_exception.EEException:
            # if it's an image, much simpler
            image = ee.Image(local_dataset_name).select(variable_name)
            if remap_value_list is not None:
                # first reduce the by date so we don't have a bunch of images
                image = image.remap(
                    remap_value_list, [1]*len(remap_value_list)).unmask(0).rename(variable_name)
        for i, points_chunk in enumerate(chunk_points(point_list, 5000)):
            try:

                circles = ee.FeatureCollection(points_chunk).map(
                    lambda feature: feature.buffer(scale)
                    if scale > 0 else feature)
                nominal_scale = image.projection().nominalScale()
                samples = image.toFloat().reduceRegions(
                    collection=circles,
                    reducer=aggregate_function,
                    scale=nominal_scale.getInfo())
                result_samples_by_year[year].extend([
                    (point['geometry']['coordinates'],
                     'NA'
                     if aggregate_function not in sample['properties']
                     else sample['properties'][aggregate_function])
                    for point, sample in
                    zip(ee.List(points_chunk).getInfo(),
                        samples.getInfo()['features'])])
                LOGGER.debug(f'processed batch {i} on {dataset_name} {variable_name} {year}')
            except Exception:
                LOGGER.exception(f'big error on {dataset_name} {variable_name} {year}')
                # fill in an NA
                result_samples_by_year[year].extend([
                    (point['geometry']['coordinates'], 'NA')
                    for point in
                    ee.List(points_chunk).getInfo()])

    return result_samples_by_year


def chunk_points(point_list, chunk_size):
    """Yield successive chunks from point_list."""
    for i in range(0, len(point_list), chunk_size):
        yield point_list[i:i + chunk_size]


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='sample points on GEE data')
    parser.add_argument('--authenticate', action='store_true', help='Pass this flag if you need to reauthenticate with GEE')
    parser.add_argument('--dataset_table_path', required=True, help='path to data table')
    parser.add_argument('--dataset_name_field', required=True, help='name of the GEE dataset field in the dataaset table path')
    parser.add_argument('--variable_name_field', required=True, help='name of the GEE variable name field in the dataaset table path')
    parser.add_argument('--scale_field', required=True, help='name of the scale of sampling field in the data table')
    parser.add_argument('--aggregate_function_field', required=True, help='name of the aggregating function field in the data table')
    parser.add_argument('--julian_range_field', required=True, help='name of field for julian day range')
    parser.add_argument('--valid_year_field', required=True, help='field for list of valid years in [y1,y2,...] format')
    parser.add_argument('--n_dataset_rows', type=int, help='limit csv read to this many rows')
    parser.add_argument(
        '--point_table_path', help='path to point sample locations',
        required=True)
    parser.add_argument('--year_field', required=True, help='field name in point table path for year')
    parser.add_argument('--long_field', required=True, help='field name in point table path for longitude')
    parser.add_argument('--lat_field', required=True, help='field name in point table path for latitude')
    parser.add_argument('--n_point_rows', type=int, help='limit csv read to this many rows')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon

    args = parser.parse_args()

    if args.authenticate:
        ee.Authenticate()
    ee.Initialize()

    point_table = pandas.read_csv(
        args.point_table_path,
        skip_blank_lines=True,
        converters={
            args.long_field: lambda x: None if x == '' else float(x),
            args.lat_field: lambda x: None if x == '' else float(x),
        },
        nrows=args.n_point_rows).dropna(how='all')

    # Explode out the YYYY-YYYY column to individual rows
    point_table = pandas.DataFrame(point_table.apply(
        expand_year_range(args.year_field), axis=1).sum())
    point_table[args.year_field] = point_table[args.year_field].astype(int)

    LOGGER.info(f'loaded {args.point_table_path}')

    dataset_table = pandas.read_csv(
        args.dataset_table_path,
        skip_blank_lines=True,
        converters={
            args.scale_field: lambda x: None if x == '' else float(x),
        },
        nrows=args.n_dataset_rows).dropna(how='all')

    LOGGER.info(f'loaded {args.dataset_table_path}')

    # Create a ThreadPoolExecutor
    futures_work_list = []

    point_features_by_year = collections.defaultdict(list)
    for year, lon, lat in zip(
            point_table[args.year_field],
            point_table[args.long_field],
            point_table[args.lat_field]):
        point_features_by_year[year].append(
            ee.Feature(ee.Geometry.Point([lon, lat], 'EPSG:4326')))

    key_set = set()
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor.
        for _, dataset_row in dataset_table.iterrows():
            # futures.append(executor.submit(get_dataset_info, dataset_row, args.dataset_name_field, args.variable_name_field))
            valid_year_list = dataset_row[args.valid_year_field]
            if isinstance(valid_year_list, str):
                valid_year_list = eval(valid_year_list)
            else:
                valid_year_list = None
            key = (
                f'{dataset_row[args.dataset_name_field]}_'
                f'{dataset_row[args.variable_name_field]}_'
                f'{dataset_row[args.aggregate_function_field]}_'
                f'{dataset_row[args.scale_field]}_'
                f'{dataset_row[args.julian_range_field]}')
            futures_work_list.append(
                (key, executor.submit(
                    sample_dataset,
                    dataset_row[args.dataset_name_field],
                    dataset_row[args.variable_name_field],
                    dataset_row[args.scale_field],
                    dataset_row[args.julian_range_field],
                    dataset_row[args.aggregate_function_field],
                    valid_year_list,
                    point_features_by_year)))
            key_set.add(key)

        # Iterate over the futures as they complete (as_completed returns them in the order they complete).

        result_dict = collections.defaultdict(dict)
        for variable_id, future in futures_work_list:
            LOGGER.debug(f'fetching {variable_id} result')
            result = future.result()  # This gets the return value from get_dataset_info when it is done.
            for year, point_list in result.items():
                for point in point_list:
                    key = (year, tuple(point[0]))
                    result_dict[key][variable_id] = point[1]
    pandas_dict_list = []
    for (year, (longitude, latitude)), value_dict in result_dict.items():
        value_dict[args.year_field] = year
        value_dict[args.long_field] = longitude
        value_dict[args.lat_field] = latitude
        pandas_dict_list.append(value_dict)
    df = pandas.DataFrame(pandas_dict_list)
    first_columns = [args.year_field, args.long_field, args.lat_field]
    other_columns = [col for col in df.columns if col not in first_columns]
    # Combine the lists to get the final column order
    final_column_order = first_columns + other_columns
    df.to_csv('output.csv', columns=final_column_order, index=False)


if __name__ == '__main__':
    main()
