"""Sample GEE datasets given pest control CSV."""
from datetime import datetime
import argparse
import configparser
import glob
import logging
import os
import sys

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

INI_DIR = './dataset_defns'

EXPECTED_INI_ELEMENTS = {
    'gee_dataset',
    'band_name',
    'valid_years',
    'filter_by',
}
EXPECTED_INI_SECTIONS = {
    'mask_types'
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
        image_only = 'image_only' in dataset_info and dataset_info['image_only']
        if image_only:
            imagecollection = ee.Image(dataset_info['gee_dataset'])
        else:
            imagecollection = ee.ImageCollection(dataset_info['gee_dataset'])

        LOGGER.debug(f"query {dataset_info['band_name']}, {year}")
        closest_year = _get_closest_num(dataset_info['valid_years'], year)
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
            # LOGGER.debug(mask_dict[mask_type].getInfo())
            # task = ee.batch.Export.image.toAsset(**{
            #     'image': mask_dict[mask_type],
            #     'description': f'{dataset_info["band_name"]}_{mask_type}',
            #     'assetId': f'projects/ecoshard-202922/assets/v1_{dataset_info["band_name"]}_{mask_type}',
            #     'scale': 30,
            #     'maxPixels': 1e13,
            #     'region': [-125, 34, -115, 40],
            #     #'region': region,
            # })
            # task.start()

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


def _sample_pheno(pts_by_year, buffer, sample_scale, datasets, datasets_to_process):
    """Sample phenology variables from https://docs.google.com/spreadsheets/d/1nbmCKwIG29PF6Un3vN6mQGgFSWG_vhB6eky7wVqVwPo

    Args:
        pts_by_year (dict): dictionary of FeatureCollection of points indexed
            by year, these are the points that are used to sample underlying
            datasets.
        buffer (float): buffer size of sample points in m
        sample_scale (float): sample size in m to treat underlying pixels
        datasets (dict): mapping of dataset id -> dataset info
        datasets_to_process (list): list of ids in ``datasets`` to process

    Returns:
        header_fields (list): list of fields to put in a CSV table
            corresponding to sampled bands
        sample_list (list): samples indexed by header fields corresponding
            to the individual points in ``pts_by_year``.
    """
    # these variables are measured in days since 1-1-1970
    LOGGER.debug('starting phenological sampling')
    julian_day_variables = [
        'Greenup_1',
        'MidGreenup_1',
        'Peak_1',
        'Maturity_1',
        'MidGreendown_1',
        'Senescence_1',
        'Dormancy_1',
    ]

    # these variables are direct quantities
    raw_variables = [
        'EVI_Minimum_1',
        'EVI_Amplitude_1',
        'EVI_Area_1',
        'QA_Overall_1',
    ]

    epoch_date = datetime.strptime('1970-01-01', "%Y-%m-%d")
    modis_phen = ee.ImageCollection(MODIS_DATASET_NAME)

    sample_list = []
    header_fields = julian_day_variables + raw_variables
    for year in pts_by_year.keys():
        LOGGER.debug(f'processing year {year}')
        year_points = pts_by_year[year]

        LOGGER.info(f'parse out MODIS variables for year {year}')
        modis_band_stack = None
        if VALID_MODIS_RANGE[0] <= year <= VALID_MODIS_RANGE[1]:
            LOGGER.debug(f'modis year: {year}')
            current_year = datetime.strptime(
                f'{year}-01-01', "%Y-%m-%d")
            days_since_epoch = (current_year - epoch_date).days
            modis_band_names = julian_day_variables + raw_variables
            bands_since_1970 = modis_phen.select(
                julian_day_variables).filterDate(
                f'{year}-01-01', f'{year}-12-31')
            julian_day_bands = (
                bands_since_1970.toBands()).subtract(days_since_epoch)
            julian_day_bands = julian_day_bands.rename(julian_day_variables)
            raw_variable_bands = modis_phen.select(
                raw_variables).filterDate(
                f'{year}-01-01', f'{year}-12-31').toBands()

            raw_variable_bands = raw_variable_bands.rename(raw_variables)
            modis_band_stack = julian_day_bands.addBands(raw_variable_bands)

            all_bands = modis_band_stack

        else:
            all_bands = ee.Image().rename(GEE_BUG_WORKAROUND_BANDNAME)
            all_bands = all_bands.addBands(ee.Image(1).rename('modis_invalid_year'))

        for dataset_id in datasets_to_process:
            mask_map, nearest_year_image = build_landcover_masks(
                year, datasets[dataset_id])
            nearest_year_image = nearest_year_image.rename(f'{dataset_id}-nearest_year')
            for mask_id, mask_image in mask_map.items():
                if modis_band_stack is not None:
                    masked_modis_band_stack = modis_band_stack.updateMask(
                        mask_image).rename([
                            f'{band_name}-{dataset_id}-{mask_id}' for band_name in modis_band_names])
                    all_bands = all_bands.addBands(masked_modis_band_stack)
                    # get modis mask
                    modis_mask = modis_band_stack.select(modis_band_stack.bandNames().getInfo()[0]).mask()
                    modis_overlap_mask = modis_mask.And(mask_image)
                    all_bands = all_bands.addBands(modis_overlap_mask.rename(f'{dataset_id}-{mask_id}-valid-modis-overlap-prop'))
                all_bands = all_bands.addBands(mask_image.rename(f'{dataset_id}-{mask_id}-pixel-prop'))

            all_bands = all_bands.addBands(nearest_year_image)

        samples = all_bands.reduceRegions(**{
            'collection': year_points,
            'reducer': REDUCER,
            'scale': sample_scale,
        }).getInfo()

        # task = ee.batch.Export.image.toAsset(**{
        #     'image': all_bands,
        #     'description': 'allbands',
        #     'assetId': 'projects/ecoshard-202922/assets/allbands',
        #     'scale': SAMPLE_SCALE,
        #     'maxPixels': 1e13,
        #     'region': year_points.first().getInfo()['geometry']['coordinates'],
        #     #'region': region,
        # })
        # task.start()

        sample_list.extend(samples['features'])
        local_header_fields = [
            x['id'] for x in all_bands.getInfo()['bands']
            if x['id'] not in header_fields and
            x['id'] != GEE_BUG_WORKAROUND_BANDNAME]
        header_fields += local_header_fields

    return header_fields, sample_list


def parse_ini(ini_path):
    """Parse ini and return a validated config."""
    basename = os.path.splitext(os.path.basename(ini_path))[0]
    dataset_config = configparser.ConfigParser(allow_no_value=True)
    dataset_config.read(ini_path)
    dataset_result = {}
    if basename not in dataset_config:
        raise ValueError(f'expected a section called {basename} but only found {dataset_config.sections()}')
    for element_id in EXPECTED_INI_ELEMENTS:
        if element_id not in dataset_config[basename]:
            raise ValueError(f'expected an entry called {element_id} but only found {dataset_config[basename].items()}')
        dataset_result[element_id] = dataset_config[basename][element_id]
    for section_id in EXPECTED_INI_SECTIONS:
        if section_id not in dataset_config:
            raise ValueError(f'expected a section called {section_id} but only found {dataset_config.sections()}')
        dataset_result[section_id] = {}
        for element_id in dataset_config[section_id].items():
            LOGGER.debug(element_id)
            dataset_result[section_id][element_id[0]] = eval(element_id[1])
    return dataset_result


def main():
    """Entry point."""
    datasets = {}
    for ini_path in glob.glob(os.path.join(INI_DIR, '*.ini')):
        dataset_config = parse_ini(ini_path)
        basename = os.path.basename(os.path.splitext(ini_path)[0])
        datasets[basename] = dataset_config

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

    args = parser.parse_args()

    default_config = configparser.ConfigParser(allow_no_value=True)
    default_config.read('global_config.ini')

    datasets_to_process = [
        x.split('dataset_')[1]
        for x in vars(args)
        if x.startswith('dataset') and vars(args)[x]]
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
        pts_by_year, args.buffer, args.sample_scale, datasets,
        datasets_to_process)

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
