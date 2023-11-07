"""Sample GEE datasets given pest control CSV."""
import argparse
import logging
import os
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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='sample points on GEE data')
    parser.add_argument('--authenticate', action='store_true', help='Pass this flag if you need to reauthenticate with GEE')
    parser.add_argument(
        '--point_table_path', help='path to point sample locations',
        required=True)
    parser.add_argument('--year_field', required=True, help='field name in point table path for year')
    parser.add_argument('--long_field', required=True, help='field name in point table path for longitude')
    parser.add_argument('--lat_field', required=True, help='field name in point table path for latitude')
    parser.add_argument('--dataset_table_path', required=True, help='path to data table')
    parser.add_argument('--dataset_name_field', required=True, help='name of the GEE dataset field in the dataaset table path')
    parser.add_argument('--variable_name_field', required=True, help='name of the GEE variable name field in the dataaset table path')
    parser.add_argument('--scale_field', required=True, help='name of the scale of sampling field in the data table')
    parser.add_argument('--aggregate_function_field', required=True, help='name of the aggregating function field in the data table')
    parser.add_argument('--julian_range_field', required=True, help='name of field for julian day range')

    parser.add_argument('--n_point_rows', type=int, help='limit csv read to this many rows')
    parser.add_argument('--n_dataset_rows', type=int, help='limit csv read to this many rows')
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
            args.year_field: lambda x: None if x == '' else int(x),
        },
        nrows=args.n_point_rows).dropna(how='all')
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
    futures = []
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor.
        for _, dataset_row in dataset_table.iterrows():
            futures.append(executor.submit(get_dataset_info, dataset_row, args.dataset_name_field, args.variable_name_field))

        # Iterate over the futures as they complete (as_completed returns them in the order they complete).
        for future in as_completed(futures):
            info = future.result()  # This gets the return value from get_dataset_info when it is done.
            LOGGER.debug(info[1:])


if __name__ == '__main__':
    main()
