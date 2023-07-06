"""Sample GEE datasets given pest control CSV."""
from datetime import datetime
import argparse
import logging
import os
import sys


from docker_contexts.backend.api import get_datasets
from docker_contexts.backend.api import _sample_pheno
import ee
import numpy
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

EXPECTED_INI_SECTIONS = {
    'mask_types', PRECIP_SECTION
}

GEE_BUG_WORKAROUND_BANDNAME = 'gee_bug_single_band_doesnt_get_name'

REDUCER = 'mean'

POLY_IN_FIELD = 'POLY-in'
POLY_OUT_FIELD = 'POLY-out'

MODIS_DATASET_NAME = 'MODIS/006/MCD12Q2'  # 500m resolution
VALID_MODIS_RANGE = (2001, 2019)


def build_landcover_masks(year, dataset_info):
    """Build landcover type masks and nearest year calculation.

    Args:
        year (int): year to build masks for
        dataset_info (dict): a map of 'gee_dataset', 'band_name',
            'valid_years', 'filter_by', and 'mask_types'->(
                dict of unique mask names mapped to list of tuple/ints)

    return dataset_map, nearest_year_image
        (dataset map maps 'mask_types' ids to binary ee.Images used
         in updatemask)
"""
    LOGGER.debug(dataset_info)
    try:
        closest_year = _get_closest_num(dataset_info['valid_years'], year)
        image_only = 'image_only' in dataset_info and dataset_info['image_only']
        gee_dataset_path = dataset_info['gee_dataset']
        if dataset_info['filter_by'] == 'dataset_year_pattern':
            gee_dataset_path = gee_dataset_path.format(year=closest_year)
        LOGGER.debug(f'****************** {gee_dataset_path}')
        if image_only:
            imagecollection = ee.Image(gee_dataset_path)
        else:
            imagecollection = ee.ImageCollection(gee_dataset_path)

        LOGGER.debug(f"query {dataset_info['band_name']}, {closest_year}({year}){dataset_info}")
        closest_year_image = ee.Image(closest_year)
        if dataset_info['filter_by'] == 'date':
            dataset = imagecollection.filter(
                ee.Filter.date(f'{closest_year}-01-01', f'{closest_year}-12-31')).first()
        elif dataset_info['filter_by'] == 'system':
            dataset = imagecollection.filter(
                ee.Filter.eq('system:index', str(closest_year))).first()
        elif not image_only:
            dataset = imagecollection.first()
        else:
            dataset = imagecollection
        band = dataset.select(dataset_info['band_name'])
        band.getInfo()
        mask_dict = {}
        if 'mask_types' in dataset_info:
            for mask_type in dataset_info['mask_types']:
                mask_dict[mask_type] = None
                for code_value in dataset_info['mask_types'][mask_type]:
                    LOGGER.debug(f'************ {mask_type} {code_value}')

                    if isinstance(code_value, tuple):
                        local_mask = (band.gte(code_value[0])).And(band.lte(code_value[1]))
                        LOGGER.debug(local_mask.getInfo())

                    else:
                        local_mask = band.eq(code_value)
                    if mask_dict[mask_type] is None:
                        mask_dict[mask_type] = local_mask
                    else:
                        mask_dict[mask_type] = mask_dict[mask_type].Or(local_mask)

        return mask_dict, closest_year_image
    except Exception:
        LOGGER.exception(f"ERROR ON {dataset_info['gee_dataset']} {dataset_info['band_name']}, {year}")
        sys.exit()


def _get_closest_num(number_list, candidate):
    """Return closest number in sorted list."""
    if isinstance(number_list, str):
        number_list = eval(number_list)
    index = (numpy.abs(numpy.array(number_list) - candidate)).argmin()
    return int(number_list[index])


# def _sample_pheno(
#         pts_by_year, buffer, sample_scale, datasets, datasets_to_process,
#         cmd_args):
#     """Sample phenology variables from https://docs.google.com/spreadsheets/d/1nbmCKwIG29PF6Un3vN6mQGgFSWG_vhB6eky7wVqVwPo

#     Args:
#         pts_by_year (dict): dictionary of FeatureCollection of points indexed
#             by year, these are the points that are used to sample underlying
#             datasets.
#         buffer (float): buffer size of sample points in m
#         sample_scale (float): sample size in m to treat underlying pixels
#         datasets (dict): mapping of dataset id -> dataset info
#         datasets_to_process (list): list of ids in ``datasets`` to process
#         cmd_args (parseargs): command line arguments used to start process

#     Returns:
#         header_fields (list): list of fields to put in a CSV table
#             corresponding to sampled bands
#         sample_list (list): samples indexed by header fields corresponding
#             to the individual points in ``pts_by_year``.
#     """
#     # these variables are measured in days since 1-1-1970
#     LOGGER.debug('starting phenological sampling')
#     julian_day_variables = [
#         'Greenup_1',
#         'MidGreenup_1',
#         'Peak_1',
#         'Maturity_1',
#         'MidGreendown_1',
#         'Senescence_1',
#         'Dormancy_1',
#     ]

#     # these variables are direct quantities
#     raw_variables = [
#         'EVI_Minimum_1',
#         'EVI_Amplitude_1',
#         'EVI_Area_1',
#         'QA_Overall_1',
#     ]

#     epoch_date = datetime.strptime('1970-01-01', "%Y-%m-%d")
#     modis_phen = ee.ImageCollection(MODIS_DATASET_NAME)

#     sample_list = []
#     header_fields = julian_day_variables + raw_variables
#     for year in pts_by_year.keys():
#         LOGGER.debug(f'processing year {year}')
#         year_points = pts_by_year[year]

#         LOGGER.info(f'parse out MODIS variables for year {year}')
#         raw_band_stack = None
#         raw_band_names = []
#         valid_modis_year = False
#         if VALID_MODIS_RANGE[0] <= year <= VALID_MODIS_RANGE[1]:
#             valid_modis_year = True
#             LOGGER.debug(f'modis year: {year}')
#             current_year = datetime.strptime(
#                 f'{year}-01-01', "%Y-%m-%d")
#             days_since_epoch = (current_year - epoch_date).days
#             raw_band_names.extend(julian_day_variables + raw_variables)
#             bands_since_1970 = modis_phen.select(
#                 julian_day_variables).filterDate(
#                 f'{year}-01-01', f'{year}-12-31')
#             julian_day_bands = (
#                 bands_since_1970.toBands()).subtract(days_since_epoch)
#             julian_day_bands = julian_day_bands.rename(julian_day_variables)
#             raw_variable_bands = modis_phen.select(
#                 raw_variables).filterDate(
#                 f'{year}-01-01', f'{year}-12-31').toBands()

#             raw_variable_bands = raw_variable_bands.rename(raw_variables)
#             raw_band_stack = julian_day_bands.addBands(raw_variable_bands)

#             all_bands = raw_band_stack
#             all_bands = all_bands.addBands(ee.Image(1).rename(
#                 'valid_modis_year'))

#         else:
#             all_bands = ee.Image().rename(GEE_BUG_WORKAROUND_BANDNAME)
#             all_bands = all_bands.addBands(ee.Image(0).rename(
#                 'valid_modis_year'))

#         for precip_dataset_id in datasets_to_process:
#             if not precip_dataset_id.startswith('precip_'):
#                 continue
#             precip_dataset = ee.ImageCollection(
#                 datasets[precip_dataset_id]['gee_dataset']).select(
#                 datasets[precip_dataset_id]['band_name'])

#             start_day, end_day = cmd_args.precip_season_start_end
#             agg_days = cmd_args.precip_aggregation_days
#             current_day = start_day
#             start_date = ee.Date(f'{year}-01-01').advance(start_day, 'day')
#             end_date = ee.Date(f'{year}-01-01').advance(end_day, 'day')
#             total_precip_bandname = (
#                 f'{precip_dataset_id}_{start_day}_{end_day}')
#             raw_band_names.append(total_precip_bandname)
#             precip_band = precip_dataset.filterDate(
#                 start_date, end_date).reduce('sum').rename(
#                 total_precip_bandname)

#             while True:
#                 if current_day >= end_day:
#                     break
#                 LOGGER.debug(f'{current_day} to {end_day}')
#                 # advance agg days - 1 since end is inclusive
#                 # (1 day is just current day not today and tomorrow)
#                 if agg_days + current_day > end_day:
#                     agg_days = end_day-current_day
#                 end_date = start_date.advance(agg_days, 'day')
#                 period_precip_bandname = f'''{precip_dataset_id}_{
#                     current_day}_{current_day+agg_days}'''
#                 raw_band_names.append(period_precip_bandname)
#                 period_precip_sample = precip_dataset.select(
#                     datasets[precip_dataset_id]['band_name']).filterDate(
#                     start_date, end_date).reduce('sum').rename(
#                     period_precip_bandname)
#                 precip_band = precip_band.addBands(period_precip_sample)
#                 start_date = end_date
#                 current_day += agg_days

#             LOGGER.debug('adding all precip bands to modis')
#             all_bands = all_bands.addBands(precip_band)
#             if raw_band_stack is not None:
#                 raw_band_stack = raw_band_stack.addBands(precip_band)
#             else:
#                 raw_band_stack = precip_band

#         for dataset_id in datasets_to_process:
#             LOGGER.debug(f'masking {dataset_id}')
#             mask_map, nearest_year_image = build_landcover_masks(
#                 year, datasets[dataset_id])
#             nearest_year_image = nearest_year_image.rename(
#                 f'{dataset_id}-nearest_year')
#             for mask_id, mask_image in mask_map.items():
#                 if raw_band_stack is not None:
#                     masked_raw_band_stack = raw_band_stack.updateMask(
#                         mask_image).rename([
#                             f'{band_name}-{dataset_id}-{mask_id}'
#                             for band_name in raw_band_names])
#                     all_bands = all_bands.addBands(masked_raw_band_stack)
#                     # get modis mask
#                     if valid_modis_year:
#                         modis_mask = raw_band_stack.select(
#                             raw_band_stack.bandNames().getInfo()[0]).mask()
#                         modis_overlap_mask = modis_mask.And(mask_image)
#                         all_bands = all_bands.addBands(
#                             modis_overlap_mask.rename(
#                                 f'''{dataset_id}-{
#                                     mask_id}-valid-modis-overlap-prop'''))

#                 all_bands = all_bands.addBands(mask_image.rename(
#                     f'{dataset_id}-{mask_id}-pixel-prop'))

#             all_bands = all_bands.addBands(nearest_year_image)

#         samples = all_bands.reduceRegions(**{
#             'collection': year_points,
#             'reducer': REDUCER,
#             'scale': sample_scale,
#         }).getInfo()

#         sample_list.extend(samples['features'])
#         local_header_fields = [
#             x['id'] for x in all_bands.getInfo()['bands']
#             if x['id'] not in header_fields and
#             x['id'] != GEE_BUG_WORKAROUND_BANDNAME]
#         header_fields += local_header_fields

#     return header_fields, sample_list


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
    table = table.dropna()

    pts_by_year = {}
    for year in table[args.year_field].unique():
        pts_by_year[year] = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point(row[args.long_field], row[args.lat_field]).buffer(args.buffer).bounds(),
                row.to_dict())
            for index, row in table[
                table[args.year_field] == year].dropna().iterrows()])

    LOGGER.debug('calculating pheno variables')
    header_fields, sample_list = _sample_pheno(
        pts_by_year, args.buffer, args.sample_scale, datasets, datasets_to_process, args)

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
