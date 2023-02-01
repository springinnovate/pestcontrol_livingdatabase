import configparser
from datetime import datetime
import os
import glob
import sys
from io import StringIO
import logging

from flask import Flask
from flask import request
from shapely.geometry import MultiPoint
import ee
import pandas as pd
import numpy


app = Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)


LOGGER = logging.getLogger()
ARGS_DATASETS = {}


def create_app(config=None):
    """Create the Geoserver STAC Flask app."""
    LOGGER.debug('starting up!')

    for ini_path in glob.glob(os.path.join(INI_DIR, '*.ini')):
        dataset_config = parse_ini(ini_path)
        basename = os.path.basename(os.path.splitext(ini_path)[0])
        ARGS_DATASETS[basename] = dataset_config

    gee_key_path = os.environ['GEE_KEY_PATH']
    credentials = ee.ServiceAccountCredentials(None, gee_key_path)
    ee.Initialize(credentials)

    app = Flask(__name__)

    @app.route('/time')
    def get_current_time():
        return {'time': datetime.now().ctime()}

    @app.route('/available_datasets')
    def get_available_datasets():
        return ARGS_DATASETS

    @app.route('/uploadfile', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            print(f'request: {request.files}')
            print(f'request.form: {request.form}')
            raw_data = request.files['file'].read().decode('utf-8')
            print(raw_data)
            long_field = request.form['long_field']
            lat_field = request.form['lat_field']
            year_field = request.form['year_field']
            buffer_size = float(request.form['buffer_size'])
            datasets_to_process = request.form['datasets_to_process'].split(',')
            table = pd.read_csv(
                StringIO(raw_data),
                skip_blank_lines=True,
                converters={
                    long_field: lambda x: None if x == '' else float(x),
                    lat_field: lambda x: None if x == '' else float(x),
                    year_field: lambda x: None if x == '' else int(x),
                },)
            point_list = [
                (row[1][0], row[1][1]) for row in table[
                    [lat_field, long_field]].iterrows()]
            points = MultiPoint(point_list)
            fields = list(table.columns)
            LOGGER.debug(f'fields: {fields}')
            f = {
              'center': [points.centroid.x, points.centroid.y],
              'data': request.files['file'].read().decode('utf-8'),
              'points': [(index, x, y) for index, (x, y) in enumerate(point_list)],
              'info': fields,
              }

            table = table.dropna()

            pts_by_year = {}
            for year in table[year_field].unique():
                pts_by_year[year] = ee.FeatureCollection([
                    ee.Feature(
                        ee.Geometry.Point(row[long_field], row[lat_field]).buffer(buffer_size).bounds(),
                        row.to_dict())
                    for index, row in table[
                        table[year_field] == year].dropna().iterrows()])

            LOGGER.debug('calculating pheno variables')
            sample_scale = 1000
            args = {}
            header_fields, sample_list = _sample_pheno(
                pts_by_year, buffer_size, sample_scale, ARGS_DATASETS, datasets_to_process, args)
            landcover_substring = '_'.join(datasets_to_process)
            file_basename = os.path.basename(request.files['file'].filename)
            csv_filename = f'sampled_{buffer_size}m_{landcover_substring}_{file_basename}'
            csv_blob_result = ''
            csv_blob_result += (
                ','.join(table.columns) + f',{",".join(header_fields)}\n')
            for sample in sample_list:
                csv_blob_result += (','.join([
                    str(sample['properties'][key])
                    for key in table.columns]) + ',')
                csv_blob_result += (','.join([
                    'invalid' if field not in sample['properties']
                    else str(sample['properties'][field])
                    for field in header_fields]) + '\n')
            f['csv_blob_result'] = csv_blob_result
            f['csv_filename'] = csv_filename
            return f
            # path = secure_filename(f.filename)
            # print(path)
            # f.save(path)
            # return 'file uploaded successfully'

    def index():
        return ARGS_DATASETS

    # wait for API calls
    #app = Flask(__name__, instance_relative_config=False)
    # app.wsgi_app = ReverseProxied(app.wsgi_app)
    # flask_cors.CORS(app)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    return app


INI_DIR = './dataset_defns'

EXPECTED_INI_ELEMENTS = {
    'gee_dataset',
    'band_name',
    'valid_years',
    'filter_by',
}
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
gd
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


def _sample_pheno(pts_by_year, buffer, sample_scale, datasets, datasets_to_process, cmd_args):
    """Sample phenology variables from https://docs.google.com/spreadsheets/d/1nbmCKwIG29PF6Un3vN6mQGgFSWG_vhB6eky7wVqVwPo

    Args:
        pts_by_year (dict): dictionary of FeatureCollection of points indexed
            by year, these are the points that are used to sample underlying
            datasets.
        buffer (float): buffer size of sample points in m
        sample_scale (float): sample size in m to treat underlying pixels
        datasets (dict): mapping of dataset id -> dataset info
        datasets_to_process (list): list of ids in ``datasets`` to process
        cmd_args (parseargs): command line arguments used to start process

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
        raw_band_stack = None
        raw_band_names = []
        valid_modis_year = False
        if VALID_MODIS_RANGE[0] <= year <= VALID_MODIS_RANGE[1]:
            valid_modis_year = True
            LOGGER.debug(f'modis year: {year}')
            current_year = datetime.strptime(
                f'{year}-01-01', "%Y-%m-%d")
            days_since_epoch = (current_year - epoch_date).days
            raw_band_names.extend(julian_day_variables + raw_variables)
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
            raw_band_stack = julian_day_bands.addBands(raw_variable_bands)

            all_bands = raw_band_stack
            all_bands = all_bands.addBands(ee.Image(1).rename('valid_modis_year'))

        else:
            all_bands = ee.Image().rename(GEE_BUG_WORKAROUND_BANDNAME)
            all_bands = all_bands.addBands(ee.Image(0).rename('valid_modis_year'))

        for precip_dataset_id in datasets_to_process:
            if not precip_dataset_id.startswith('precip_'):
                continue
            precip_dataset = ee.ImageCollection(datasets[precip_dataset_id]['gee_dataset']).select(datasets[precip_dataset_id]['band_name'])

            start_day, end_day = cmd_args.precip_season_start_end
            agg_days = cmd_args.precip_aggregation_days
            current_day = start_day
            start_date = ee.Date(f'{year}-01-01').advance(start_day, 'day')
            end_date = ee.Date(f'{year}-01-01').advance(end_day, 'day')
            total_precip_bandname = f'{precip_dataset_id}_{start_day}_{end_day}'
            raw_band_names.append(total_precip_bandname)
            precip_band = precip_dataset.filterDate(
                start_date, end_date).reduce('sum').rename(total_precip_bandname)

            while True:
                if current_day >= end_day:
                    break
                LOGGER.debug(f'{current_day} to {end_day}')
                # advance agg days - 1 since end is inclusive (1 day is just current day not today and tomorrow)
                if agg_days + current_day > end_day:
                    agg_days = end_day-current_day
                end_date = start_date.advance(agg_days, 'day')
                period_precip_bandname = f'{precip_dataset_id}_{current_day}_{current_day+agg_days}'
                raw_band_names.append(period_precip_bandname)
                period_precip_sample = precip_dataset.select(datasets[precip_dataset_id]['band_name']).filterDate(
                    start_date, end_date).reduce('sum').rename(period_precip_bandname)
                precip_band = precip_band.addBands(period_precip_sample)
                start_date = end_date
                current_day += agg_days

            LOGGER.debug('adding all precip bands to modis')
            all_bands = all_bands.addBands(precip_band)
            if raw_band_stack is not None:
                raw_band_stack = raw_band_stack.addBands(precip_band)
            else:
                raw_band_stack = precip_band

        for dataset_id in datasets_to_process:
            LOGGER.debug(f'masking {dataset_id}')
            mask_map, nearest_year_image = build_landcover_masks(
                year, datasets[dataset_id])
            nearest_year_image = nearest_year_image.rename(f'{dataset_id}-nearest_year')
            for mask_id, mask_image in mask_map.items():
                if raw_band_stack is not None:
                    masked_raw_band_stack = raw_band_stack.updateMask(
                        mask_image).rename([
                            f'{band_name}-{dataset_id}-{mask_id}' for band_name in raw_band_names])
                    all_bands = all_bands.addBands(masked_raw_band_stack)
                    # get modis mask
                    if valid_modis_year:
                        modis_mask = raw_band_stack.select(raw_band_stack.bandNames().getInfo()[0]).mask()
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
    found_expected_section = False
    for section_id in EXPECTED_INI_SECTIONS:
        if section_id in dataset_config:
            found_expected_section = True
            break
    if not found_expected_section:
        raise ValueError(
            f'expected any one of sections called {EXPECTED_INI_SECTIONS} but only found {dataset_config.sections()}')
    dataset_result[section_id] = {}
    for element_id in dataset_config[section_id].items():
        LOGGER.debug(element_id)
        dataset_result[section_id][element_id[0]] = eval(element_id[1])
    return dataset_result


if __name__ == '__main__':
    app = create_app()
    app.run()
