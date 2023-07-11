"""Sample GEE datasets given pest control CSV."""
import argparse
import logging
import os


from docker_contexts.backend.api import get_datasets
from docker_contexts.backend.api import _process_table
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


def main():
    """Entry point."""
    datasets = get_datasets()

    LOGGER.debug(datasets)
    parser = argparse.ArgumentParser(
        description='sample points on GEE data')
    parser.add_argument('csv_path', help='path to CSV data table')
    parser.add_argument('--year_field', default='crop_year', help='field name in csv_path for year, default `year_field`')
    parser.add_argument('--long_field', default='field_longitude', help='field name in csv_path for longitude, default `long_field`')
    parser.add_argument('--lat_field', default='field_latitude', help='field name in csv_path for latitude, default `lat_field`')
    parser.add_argument('--buffer', type=float, default=1000, help='buffer distance in meters around point to do aggregate analysis, default 1000m')
    parser.add_argument('--sample_scale', type=float, default=500, help='underlying pixel size to sample at, defaults to 500m (modis resolution)')
    parser.add_argument('--n_rows', type=int, help='limit csv read to this many rows')
    parser.add_argument('--authenticate', action='store_true', help='Pass this flag if you need to reauthenticate with GEE')
    for dataset_id in datasets:
        parser.add_argument(
            f'--dataset_{dataset_id}', default=False, action='store_true',
            help=f'use the {dataset_id} {datasets[dataset_id]["band_name"]} {datasets[dataset_id]["gee_dataset"]} dataset for masking')
    parser.add_argument(
        '--precip_season_start_end', type=int, nargs=2, help=(
            'Two arguments defining the START day offset of the season from number of days away from Jan 1 of the '
            'current year (can be negative) to number of days away from Jan 1 for end of season (e.x. -100, 100 '
            'defines a season starting at Sep 23 of previous year to April 10 of current year.'))
    parser.add_argument(
        '--precip_aggregation_days', type=int, help='number of days to aggregate precipitation over to report in periods.')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon

    args = parser.parse_args()

    datasets_to_process = [
        x.split('dataset_')[1]
        for x in vars(args)
        if x.startswith('dataset') and vars(args)[x]]

    precip_required = list(
        dataset_id for dataset_id in datasets_to_process
        if PRECIP_SECTION in datasets[dataset_id])

    if precip_required:
        if args.precip_season_start_end is None:
            raise ValueError(
                f'Expected --precip_season_start_end because {precip_required} is/are precip')
        if args.precip_aggregation_days is None:
            raise ValueError(
                f'Expected --precip_aggregation_days because {precip_required} is/are precip')
    landcover_substring = '_'.join(datasets_to_process)
    LOGGER.debug(landcover_substring)
    if args.authenticate:
        ee.Authenticate()
    ee.Initialize()

    LOGGER.info(f'loading {args.csv_path}')
    table = pandas.read_csv(
        args.csv_path,
        skip_blank_lines=True,
        converters={
            args.long_field: lambda x: None if x == '' else float(x),
            args.lat_field: lambda x: None if x == '' else float(x),
            args.year_field: lambda x: None if x == '' else int(x),
        },
        nrows=args.n_rows)
    LOGGER.info(f'loaded {args.csv_path}')

    # the _ ignores the URL info
    header_fields, sample_list, _ = _process_table(
        table, datasets_to_process, args.year_field, args.long_field,
        args.lat_field, args.buffer, vars(args))

    with open(f'sampled_{args.buffer}m_{landcover_substring}_{os.path.basename(args.csv_path)}', 'w') as table_file:
        table_file.write(
            ','.join(table.columns) + f',{",".join(header_fields)}\n')
        for sample in sample_list:
            table_file.write(','.join([
                str(sample['properties'][key])
                for key in table.columns]) + ',')
            table_file.write(','.join([
                'invalid' if field not in sample['properties']
                else str(sample['properties'][field])
                for field in header_fields]) + '\n')


if __name__ == '__main__':
    main()
