"""Sample GEE datasets given pest control CSV."""
from datetime import datetime
import argparse
import configparser
import glob
import json
import logging
import os
import sys

import geopandas
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

SAMPLE_SCALE = 30  # this is the raster resolution of which to sample the rasters at

REDUCER = 'mean'

POLY_IN_FIELD = 'POLY-in'
POLY_OUT_FIELD = 'POLY-out'

MODIS_DATASET_NAME = 'MODIS/006/MCD12Q2'  # 500m resolution
VALID_MODIS_RANGE = (2001, 2019)


def build_landcover_masks(year, dataset_info, ee_poly):
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

        result_dict = {}
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
        mask_dict = {mask_type: ee.Image(0) for mask_type in dataset_info['mask_types']}
        for mask_type in dataset_info['mask_types']:
            for code_value in dataset_info['mask_types'][mask_type]:
                if isinstance(code_value, tuple):
                    mask_dict[mask_type] = mask_dict[mask_type].Or(
                        band.gte(code_value[0]).And(band.lte(code_value[1])))
                else:
                    mask_dict[mask_type] = mask_dict[mask_type].Or(
                        band.eq(code_value))
            mask_dict[mask_type].getInfo()
        LOGGER.debug('debug all done')
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


def _corine_natural_cultivated_mask(year):
    """Natural: 311-423, Cultivated: 211 - 244."""
    closest_year = _get_closest_num(CORINE_VALID_YEARS, year)
    corine_imagecollection = ee.ImageCollection(CORINE_DATASET)

    corine_landcover = corine_imagecollection.filter(
        ee.Filter.eq('system:index', str(closest_year))).first().select('landcover')

    natural_mask = ee.Image(0).where(
        corine_landcover.gte(311).And(corine_landcover.lte(423)), 1)
    natural_mask = natural_mask.rename(CORINE_NATURAL_FIELD)

    cultivated_mask = ee.Image(0).where(
        corine_landcover.gte(211).And(corine_landcover.lte(244)), 1)
    cultivated_mask = cultivated_mask.rename(CORINE_CULTIVATED_FIELD)
    return natural_mask, cultivated_mask, closest_year


def _nlcd_natural_cultivated_mask(year, ee_poly):
    """Natural for NLCD in 41-74 or 90-95."""
    closest_year = _get_closest_num(NLCD_VALID_YEARS, year)
    nlcd_imagecollection = ee.ImageCollection(NLCD_DATASET)
    nlcd_year = nlcd_imagecollection.filter(
        ee.Filter.eq('system:index', str(closest_year))).first().select('landcover')
    # natural 41-74 & 90-95
    natural_mask = ee.Image(0).where(
        nlcd_year.gte(41).And(nlcd_year.lte(74)).Or(
            nlcd_year.gte(90).And(nlcd_year.lte(95))), 1)
    natural_mask = natural_mask.rename(NLCD_NATURAL_FIELD)

    cultivated_mask = ee.Image(0).where(
        nlcd_year.gte(81).And(nlcd_year.lte(82)), 1)
    cultivated_mask = cultivated_mask.rename(NLCD_CULTIVATED_FIELD)

    if not ee_poly:
        return natural_mask, cultivated_mask, closest_year

    # create masks of in/out using same bounds as base image
    polymask = natural_mask.updateMask(ee.Image(1).clip(ee_poly)).unmask().gt(0)
    inv_polymask = polymask.unmask().Not()

    natural_mask_in = natural_mask.updateMask(polymask)
    cultivated_mask_in = cultivated_mask.updateMask(polymask)
    #closest_year_in = closest_year.updateMask(polymask)
    natural_mask_out = natural_mask.updateMask(inv_polymask)
    cultivated_mask_out = cultivated_mask.updateMask(inv_polymask)
    #closest_year_out = closest_year.updateMask(inv_polymask)

    return (
        natural_mask_in, cultivated_mask_in,
        natural_mask_out, cultivated_mask_out, closest_year)


def _sample_pheno(pts_by_year, buffer, datasets, datasets_to_process, ee_poly):
    """Sample phenology variables from https://docs.google.com/spreadsheets/d/1nbmCKwIG29PF6Un3vN6mQGgFSWG_vhB6eky7wVqVwPo

    Args:
        pts_by_year (dict): dictionary of FeatureCollection of points indexed
            by year, these are the points that are used to sample underlying
            datasets.
        buffer (float): buffer size of sample points in m
        datasets (dict): mapping of dataset id -> dataset info
        datasets_to_process (list): list of ids in ``datasets`` to process
        ee_poly (ee.Polygon): if not None, additionally filter samples on the
            nlcd/corine datasets to see what's in or out.

    Returns:
        header_fields (list):

    """
    # these variables are measured in days since 1-1-1970
    LOGGER.debug('starting sample')
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
    LOGGER.debug('got modis')
    # base_header_fields = [
    #     f'MODIS-{field}'
    #     for field in julian_day_variables+raw_variables]
    # # make copy so we don't overwrite
    # header_fields = list(base_header_fields)
    # for dataset_id in datasets_to_process:
    #     LOGGER.debug(datasets[dataset_id])
    #     LOGGER.debug(datasets[dataset_id]['mask_types'])
    #     for mask_code_id in datasets[dataset_id]['mask_types']:
    #         for header_id in base_header_fields:
    #             LOGGER.debug(header_id)
    #             header_fields.append(
    #                 f'{header_id}-{dataset_id}-{mask_code_id}')
    #         if ee_poly:
    #             header_fields += [
    #                 f'{dataset_id}-{POLY_IN_FIELD}',
    #                 f'{dataset_id}-{POLY_OUT_FIELD}']

    # if ee_poly:
    #     header_fields += [POLY_IN_FIELD, POLY_OUT_FIELD]

    # LOGGER.debug(f'header fields: {header_fields}')

    sample_list = []
    all_band_names = []
    for year in pts_by_year.keys():
        LOGGER.debug(f'processing year {year}')
        year_points = pts_by_year[year]

        # if nlcd_flag:
        #     if not ee_poly:
        #         nlcd_natural_mask, nlcd_cultivated_mask, nlcd_closest_year = \
        #             _nlcd_natural_cultivated_mask(year, None)
        #     else:
        #         (nlcd_natural_mask_poly_in, nlcd_cultivated_mask_poly_in,
        #          nlcd_natural_mask_poly_out, nlcd_cultivated_mask_poly_out,
        #          nlcd_closest_year) = \
        #             _nlcd_natural_cultivated_mask(year, ee_poly)
        #     LOGGER.debug(f'nlcd_closest_year: {nlcd_closest_year}')

        # if corine_flag:
        #     corine_natural_mask, corine_cultivated_mask, corine_closest_year = \
        #         _corine_natural_cultivated_mask(year)
        LOGGER.info('parse out MODIS variables for year {year}')
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

            band_names = [x['id'] for x in julian_day_bands.getInfo()['bands']]
            LOGGER.debug(f'julian day n bands: {len(band_names)} {band_names}')

            modis_band_stack = julian_day_bands.addBands(raw_variable_bands)

            band_names = [x['id'] for x in modis_band_stack.getInfo()['bands']]
            LOGGER.debug(f'local bands n bands: {len(band_names)} {band_names}')

            all_bands = modis_band_stack

            mask_dict = {}
            for dataset_id in datasets_to_process:
                mask_map, nearest_year_image = build_landcover_masks(
                    year, datasets[dataset_id], ee_poly)
                nearest_year_image = nearest_year_image.rename(f'{dataset_id}-nearest_year')
                for mask_id, mask_image in mask_map.items():
                    band_names = [x['id'] for x in modis_band_stack.getInfo()['bands']]
                    LOGGER.debug(f'n bands: {len(band_names)} {band_names}, {len(modis_band_names)} {modis_band_names}')
                    masked_modis_band_stack = modis_band_stack.updateMask(
                        mask_image).rename([
                            f'{band_name}-{mask_id}' for band_name in modis_band_names])
                    band_names = [x['id'] for x in masked_modis_band_stack.getInfo()['bands']]
                    LOGGER.debug(f'n bands: {band_names}')

                    all_bands = all_bands.addBands(masked_modis_band_stack)
                all_bands = all_bands.addBands(nearest_year_image)

            # determine area in/out of point area
            if ee_poly:
                def area_in_out(feature):
                    feature_area = feature.area()
                    area_in = ee_poly.intersection(feature.geometry()).area()
                    return feature.set({
                        POLY_OUT_FIELD: feature_area.subtract(area_in),
                        POLY_IN_FIELD: area_in})

            year_points = year_points.map(area_in_out).getInfo()

        samples = all_bands.reduceRegions(**{
            'collection': year_points,
            'reducer': REDUCER,
            'scale': SAMPLE_SCALE}).getInfo()
        sample_list.extend(samples['features'])

    header_fields = [x['id'] for x in all_bands.getInfo()['bands']]
    LOGGER.debug(f'***********\n\n{all_bands.getInfo()}\n\n**************')
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
    parser.add_argument('--polygon_path', type=str, help='path to local polygon to sample')
    parser.add_argument('--n_rows', type=int, help='limit csv read to this many rows')
    parser.add_argument('--authenticate', action='store_true', help='Pass this flag if you need to reauthenticate with GEE')
    for dataset_id in datasets:
        parser.add_argument(
            f'--dataset_{dataset_id}', default=False, action='store_true',
            help=f'use the {dataset_id} {datasets[dataset_id]["band_name"]} {datasets[dataset_id]["gee_dataset"]} dataset for masking')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon

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

    table = pandas.read_csv(
        args.csv_path, 
        skip_blank_lines=True,
        converters={
            args.long_field: lambda x: None if x == '' else float(x),
            args.lat_field: lambda x: None if x == '' else float(x),
            args.year_field: lambda x: None if x == '' else int(x),
        },
        nrows=args.n_rows)
    table = table.dropna()
    LOGGER.debug(table)
    ee_poly = None
    if args.polygon_path:
        # convert to GEE polygon
        gp_poly = geopandas.read_file(args.polygon_path).to_crs('EPSG:4326')
        json_poly = json.loads(gp_poly.to_json())
        coords = []
        for json_feature in json_poly['features']:
            coords.append(json_feature['geometry']['coordinates'])
        ee_poly = ee.Geometry.MultiPolygon(coords)

    pts_by_year = {}
    for year in table[args.year_field].unique():
        pts_by_year[year] = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point(row[args.long_field], row[args.lat_field]).buffer(args.buffer),
                row.to_dict())
            for index, row in table[
                table[args.year_field] == year].dropna().iterrows()])

    LOGGER.debug('calculating pheno variables')
    header_fields, sample_list = _sample_pheno(
        pts_by_year, args.buffer, datasets, datasets_to_process, ee_poly)

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
