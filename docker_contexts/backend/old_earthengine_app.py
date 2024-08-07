"""Backend code to query and annotate point data from remote data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
import configparser
import glob
import logging
import os
import json
import secrets
import sys
import uuid
import time
import threading

from flask import request
from flask import session
from flask import jsonify
from flask import make_response
from shapely.geometry import MultiPoint
from flask import Flask
import ee
import numpy
import pandas as pd

from database_model_definitions import FILTERABLE_FIELDS

app = Flask(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
ARGS_DATASETS = {}

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

TASK_LOOKUP = {}
LOCAL_CONTEXT = {}


def process_file():
    try:
        LOGGER.debug('PROCESSING FILE')
        task_id = str(uuid.uuid4())  # generate a unique task id
        TASK_LOOKUP[task_id] = {
            'state': 'RUNNING',
            'result': None,
            'start_time': time.time()}
        LOGGER.debug(f'request: {request.files}')
        LOGGER.debug(f'request.form: {request.form}')
        file_data = request.files['file'].read().decode('utf-8')
        file_basename = os.path.basename(request.files['file'].filename)
        long_field = request.form['long_field']
        lat_field = request.form['lat_field']
        year_field = request.form['year_field']
        buffer_size = float(request.form['buffer_size'])
        datasets_to_process = json.loads(
            request.form['datasets_to_process'])
        threading.Thread(
            target=process_file_worker, args=(
                file_basename, file_data, long_field, lat_field, year_field,
                buffer_size, datasets_to_process, task_id)).start()
        return {
            'task_id': task_id
            }
    except Exception:
        LOGGER.exception('something bad happened on process_file')
        raise


def download_raster_worker(global_task_id, raster_id):
    try:
        LOGGER.debug('STARTING download_raster_worker')
        local_task_id = str(uuid.uuid4())  # generate a unique task id
        TASK_LOOKUP[local_task_id] = {
            'state': 'RUNNING',
            'result': None,
            'start_time': time.time()}
        threading.Thread(
            target=get_download_url,
            args=(global_task_id, local_task_id, raster_id)).start()
        return {
            'task_id': local_task_id
        }
    except Exception:
        LOGGER.exception('something bad happened on download_raster_worker')
        raise


def _process_table(
        table, datasets_to_process, year_field, long_field, lat_field,
        buffer_size, cmd_args):
    pts_by_year = {}
    n_points_per_batch = 4000
    for year in table[year_field].unique():
        year = int(year)  # Convert year to int
        year_data = table[table[year_field] == year]
        num_batches = -(-len(year_data) // n_points_per_batch) # round up

        batches = numpy.array_split(year_data, num_batches)
        pts_by_year[year] = []
        for batch in batches:
            batch_collection = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point(row[long_field], row[lat_field]).
                    buffer(buffer_size),
                    row.to_dict())
                for index, row in batch.iterrows()])

            pts_by_year[year].append(batch_collection)  # Append the batch collection to the year's list

    LOGGER.debug('calculating pheno variables')
    sample_scale = 1000
    datasets = get_datasets()
    header_fields, sample_list, band_and_bounds_by_id = _sample_pheno(
        pts_by_year, buffer_size, sample_scale, datasets,
        datasets_to_process, cmd_args)
    return header_fields, sample_list, band_and_bounds_by_id


def long_converter(x):
    if x == '':
        return None
    try:
        val = float(x)
        if val > 180 or val < -180:
            raise ValueError(f'{val} is outside the range of -180 to 180 expected by longitude.')
        return x
    except ValueError:
        raise ValueError(f"Error converting '{x}' to float for long_field")

def lat_converter(x):
    if x == '':
        return None
    try:
        val = float(x)
        if val > 90 or val < -90:
            raise ValueError(f'{val} is outside the range of -90 to 90 expected by latitude.')
        return x
    except ValueError:
        raise ValueError(f"Error converting '{x}' to float for lat_field")

def year_converter(x):
    if x == '':
        return None
    try:
        val = int(x)
        if val > 2100 or val < 1950:
            raise ValueError(f'expected a year value between 1950 and 2100 but got {val}')
        return int(x)
    except ValueError:
        raise ValueError(f"Error converting '{x}' to int for year_field")



def process_file_worker(
        file_basename, file_data, long_field, lat_field, year_field,
        buffer_size, datasets_to_process, task_id):
    LOGGER.debug('processing table file')
    try:
        input_table = pd.read_csv(
            StringIO(file_data),
            skip_blank_lines=True,
            converters={
                long_field: long_converter,
                lat_field: lat_converter,
                year_field: year_converter,
            }, low_memory=False)
        input_table = input_table.dropna(subset=[lat_field, long_field, year_field])
        input_table = input_table.fillna('')
        lat_lng_year_table = input_table[[lat_field, long_field, year_field]]
        # table[lat_field] = pd.to_numeric(table[lat_field], errors='coerce')
        # table[long_field] = pd.to_numeric(table[long_field], errors='coerce')
        point_list = [
            (row[0], row[1])
            for index, row in lat_lng_year_table[[lat_field, long_field]].iterrows()
        ]

        points = MultiPoint(point_list)
        fields = list(lat_lng_year_table.columns)
        result_payload = {
            'center': [points.centroid.x, points.centroid.y],
            'data': file_data,
            'points': point_list,
            'info': fields,
            }
        precip_args = {}
        header_fields, sample_list, band_and_bounds_by_id = _process_table(
            lat_lng_year_table, datasets_to_process,
            year_field, long_field, lat_field, buffer_size,
            precip_args)
        LOGGER.debug(sample_list)
        return
        result_payload['band_ids'] = list(band_and_bounds_by_id)

        landcover_substring = '_'.join([x['key'] for x in datasets_to_process])
        csv_filename = f'''sampled_{buffer_size}m_{landcover_substring}_{
            file_basename}'''
        csv_blob_result = ''
        csv_blob_result += (
            ','.join(lat_lng_year_table.columns) + f',{",".join(header_fields)}\n')
        geojson_str_list = []
        for sample in sample_list:
            csv_blob_result += (','.join([
                str(sample['properties'][key])
                for key in lat_lng_year_table.columns]) + ',')
            csv_blob_result += (','.join([
                'invalid' if field not in sample['properties']
                else str(sample['properties'][field])
                for field in header_fields]) + '\n')
            geojson_str_list.append(sample['geometry'])
        # join to original table here?
        pheno_table = pd.read_csv(StringIO(csv_blob_result))
        result_table = pd.merge(pheno_table, input_table, on=[lat_field, long_field, year_field], how='inner')
        table_string_io = StringIO()
        result_table.to_csv(table_string_io, index=False)
        result_payload['csv_blob_result'] = table_string_io.getvalue()
        table_string_io.close()
        result_payload['csv_filename'] = csv_filename
        result_payload['geojson_str_list'] = geojson_str_list
        LOCAL_CONTEXT[task_id] = {
            'band_and_bounds_by_id': band_and_bounds_by_id
        }
        TASK_LOOKUP[task_id].update({
            'state': 'SUCCESS',
            'result': result_payload})
    except Exception as e:
        LOGGER.exception('something bad happened on process_file')
        error_type = type(e).__name__
        error_message = str(e)
        TASK_LOOKUP[task_id].update({
            'state': 'FAILURE',
            'result': error_message
            })
        return make_response(jsonify(
            error={'error': f'{error_type}: {error_message}'}), 500)


def create_app(config=None):
    """Create the Geoserver STAC Flask app."""
    global ARGS_DATASETS
    LOGGER.info('starting up!')
    gee_key_path = os.environ['GEE_KEY_PATH']
    credentials = ee.ServiceAccountCredentials(None, gee_key_path)
    ee.Initialize(credentials)
    ARGS_DATASETS = get_datasets()

    app = Flask(__name__)
    app.secret_key = secrets.token_hex()

    @app.route('/api/set/')
    def set():
        session['key'] = 'value'
        return 'ok'

    @app.route('/api/uploadfile', methods=['POST'])
    def upload_file():
        LOGGER.debug('uploading file')
        return process_file()

    @app.route('/api/task/<task_id>')
    def get_task(task_id):
        # TODO: delete if task is complete or error
        if task_id not in TASK_LOOKUP:
            return f'{task_id} not found', 500
        TASK_LOOKUP[task_id]['time_running'] = (
            time.time()-TASK_LOOKUP[task_id]['start_time'])
        return TASK_LOOKUP[task_id]

    @app.route('/api/download_raster/<task_id>/<raster_id>', methods=['POST'])
    def download_raster(task_id, raster_id):
        # TODO: delete if task is complete or error
        if task_id not in LOCAL_CONTEXT:
            return f'{task_id} not found', 500
        return download_raster_worker(task_id, raster_id)

    @app.route('/api/get/')
    def get():
        return session.get('key', 'not set')

    @app.route('/api/time')
    def get_current_time():
        return {'time': datetime.now().ctime()}

    @app.route('/api/available_datasets')
    def get_available_datasets():
        return get_datasets()

    @app.route('/api/searchable_fields', methods=['GET'])
    def searchable_fields():
        LOGGER.debug(FILTERABLE_FIELDS)
        return {'filterable_fields': FILTERABLE_FIELDS}

    def index():
        return get_datasets()

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    return app


INI_DIR = os.path.join(os.path.dirname(__file__), 'dataset_defns')

EXPECTED_INI_ELEMENTS = {
    'gee_dataset',
    'band_name',
    'valid_years',
    'filter_by',
    'bounds',
}

OPTIONAL_INI_ELEMENTS = {
    'image_only',
    'disabled',
    'viewer',
    'documentation',
    'dataset_name',
}

SEASONAL_SECTION = 'seasonal'
AGGREGATE_FUNCTION_FIELD = 'aggregate_function'

EXPECTED_INI_SECTIONS = {
    'mask_types', SEASONAL_SECTION
}

GEE_BUG_WORKAROUND_BANDNAME = 'gee_bug_single_band_doesnt_get_name'

REDUCER = 'mean'

POLY_IN_FIELD = 'POLY-in'
POLY_OUT_FIELD = 'POLY-out'

MODIS_DATASET_NAME = 'MODIS/061/MCD12Q2'  # 500m resolution
VALID_MODIS_RANGE = (2001, 2021)


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
        image_only = (
            'image_only' in dataset_info and
            dataset_info['image_only'].lower() == 'true')
        gee_dataset_path = dataset_info['gee_dataset']
        if dataset_info['filter_by'] == 'dataset_year_pattern':
            gee_dataset_path = gee_dataset_path.format(year=closest_year)
        if image_only:
            imagecollection = ee.Image(gee_dataset_path)
        else:
            imagecollection = ee.ImageCollection(gee_dataset_path)

        LOGGER.debug(
            f"query {dataset_info['band_name']}, "
            f"{closest_year}({year}){dataset_info}")
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
        mask_dict = {}
        if 'mask_types' in dataset_info:
            for mask_type in dataset_info['mask_types']:
                mask_dict[mask_type] = None
                for code_value in dataset_info['mask_types'][mask_type]:
                    if isinstance(code_value, tuple):
                        local_mask = (band.gte(code_value[0])).And(band.lte(code_value[1]))
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


def _sample_pheno(
        pts_by_year, buffer, sample_scale, datasets,
        datasets_to_process, cmd_args):
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
            to the individual points in ``pts_by_year``.\
        url_by_header_id (dict): a url to a geotiff bounded by the points
            indexed by the header field + year of sample.
    """
    # these variables are measured in days since 1-1-1970
    executor = ThreadPoolExecutor(max_workers=10)
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

    sample_future_list = []
    if cmd_args['process_phenological_vars']:
        header_fields = julian_day_variables + raw_variables
    else:
        header_fields = []
    header_fields_set = set(header_fields)
    band_and_bounds_by_id = {}
    for year, year_point_list in pts_by_year.items():
        year = int(year)
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
            current_year_modis_phen = ee.Image(f'{MODIS_DATASET_NAME}/{year}_01_01')
            bands_since_1970 = current_year_modis_phen.select(
                julian_day_variables)
            julian_day_bands = (
                bands_since_1970).subtract(days_since_epoch)
            julian_day_bands = julian_day_bands.rename(julian_day_variables)
            raw_variable_bands = current_year_modis_phen.select(
                raw_variables)

            raw_variable_bands = raw_variable_bands.rename(raw_variables)
            raw_band_stack = julian_day_bands.addBands(raw_variable_bands)
            all_bands = raw_band_stack
            all_bands = all_bands.addBands(ee.Image(1).rename(
                'valid_modis_year'))
        else:
            all_bands = ee.Image().rename(GEE_BUG_WORKAROUND_BANDNAME)
            all_bands = all_bands.addBands(ee.Image(0).rename(
                'valid_modis_year'))
        if 'valid_modis_year' not in header_fields_set:
            header_fields.append('valid_modis_year')
            header_fields_set.add(header_fields[-1])

        for seasonal_dataset_id in datasets_to_process:
            if SEASONAL_SECTION not in datasets[seasonal_dataset_id]:
                continue
            seasonal_dataset = ee.ImageCollection(
                datasets[seasonal_dataset_id]['gee_dataset']).select(
                datasets[seasonal_dataset_id]['band_name'])
            start_day, end_day = cmd_args['season_start_end']

            agg_days = cmd_args['precip_aggregation_days']
            if agg_days is None:
                agg_days = 365
            current_day = start_day
            start_date = ee.Date(f'{year}-01-01').advance(start_day, 'day')
            end_date = ee.Date(f'{year}-01-01').advance(end_day, 'day')
            total_seasonal_bandname = (
                f'{seasonal_dataset_id}_{start_day}_{end_day}')
            raw_band_names.append(total_seasonal_bandname)
            if raw_band_names[-1] not in header_fields_set:
                header_fields.append(raw_band_names[-1])
                header_fields_set.add(header_fields[-1])

            if 'limit_to_n_samples' in cmd_args:
                mode, sample_count = cmd_args['limit_to_n_samples']
                sample_count = int(sample_count)

                sorted_dataset = seasonal_dataset.filterDate(start_date, end_date).sort(
                    'temperature', mode == 'bottom')
                _seasonal_band = sorted_dataset.limit(sample_count)
            else:
                _seasonal_band = seasonal_dataset.filterDate(
                    start_date, end_date)

            _reduce_fn = cmd_args['override_aggregate']
            if _reduce_fn is None:
                # default to the dataset default
                _reduce_fn = datasets[seasonal_dataset_id][SEASONAL_SECTION][AGGREGATE_FUNCTION_FIELD]
            seasonal_band = _seasonal_band.reduce(
                _reduce_fn).rename(
                total_seasonal_bandname)
            while True:
                if current_day >= end_day:
                    break
                LOGGER.debug(f'{current_day} to {end_day}')
                # advance agg days - 1 since end is inclusive
                # (1 day is just current day not today and tomorrow)
                if agg_days + current_day > end_day:
                    agg_days = end_day-current_day
                end_date = start_date.advance(agg_days, 'day')
                period_seasonal_bandname = f'''{seasonal_dataset_id}_{
                    current_day}_{current_day+agg_days}'''
                raw_band_names.append(period_seasonal_bandname)
                if raw_band_names[-1] not in header_fields_set:
                    header_fields.append(raw_band_names[-1])
                    header_fields_set.add(header_fields[-1])
                period_precip_sample = seasonal_dataset.select(
                    datasets[seasonal_dataset_id]['band_name']).filterDate(
                    start_date, end_date).reduce(
                        datasets[
                            seasonal_dataset_id][
                            SEASONAL_SECTION][
                            AGGREGATE_FUNCTION_FIELD]).rename(
                    period_seasonal_bandname)
                seasonal_band = seasonal_band.addBands(period_precip_sample)
                start_date = end_date
                current_day += agg_days

            LOGGER.debug('adding all precip bands to modis')
            all_bands = all_bands.addBands(seasonal_band)
            if raw_band_stack is not None:
                raw_band_stack = raw_band_stack.addBands(seasonal_band)
            else:
                raw_band_stack = seasonal_band

        if cmd_args['process_phenological_vars']:
            for dataset_id in datasets_to_process:
                mask_map, nearest_year_image = build_landcover_masks(
                    year, datasets[dataset_id])
                for mask_id, mask_image in mask_map.items():
                    if raw_band_stack is not None:
                        mask_band_names = [
                            f'{band_name}-{dataset_id}-{mask_id}'
                            for band_name in raw_band_names]
                        masked_raw_band_stack = raw_band_stack.updateMask(
                            mask_image).rename(mask_band_names)
                        for mask_band_name in mask_band_names:
                            if mask_band_name not in header_fields_set:
                                header_fields.append(mask_band_name)
                                header_fields_set.add(header_fields[-1])
                        all_bands = all_bands.addBands(masked_raw_band_stack)
                        # get modis mask
                        if valid_modis_year:
                            # just get the first image because it will have the
                            # base mask
                            modis_mask = raw_band_stack.select(0).mask()
                            modis_overlap_mask = modis_mask.And(mask_image)
                            modis_field_name = f'''{dataset_id}-{mask_id}-valid-modis-overlap-prop'''
                            if modis_field_name not in header_fields_set:
                                header_fields.append(modis_field_name)
                                header_fields_set.add(header_fields[-1])
                            all_bands = all_bands.addBands(
                                modis_overlap_mask.rename(modis_field_name))

                    pixel_prop_field_name = f'{dataset_id}-{mask_id}-pixel-prop'
                    if pixel_prop_field_name not in header_fields_set:
                        header_fields.append(pixel_prop_field_name)
                        header_fields_set.add(header_fields[-1])
                    all_bands = all_bands.addBands(
                        mask_image.rename(pixel_prop_field_name))

                nearest_year_field_name = f'{dataset_id}-nearest_year'
                if nearest_year_field_name not in header_fields_set:
                    header_fields.append(nearest_year_field_name)
                    header_fields_set.add(header_fields[-1])
                nearest_year_image = nearest_year_image.rename(nearest_year_field_name)
                all_bands = all_bands.addBands(nearest_year_image)

        for year_points in year_point_list:
            future = executor.submit(
                _process_sample_regions, all_bands, year_points, sample_scale)
            sample_future_list.append(future)

        # Assuming collections is your list of ee.FeatureCollection objects
        combined_geometry = ee.Geometry.MultiPolygon([])
        for collection in year_point_list:
            collection_geometry = collection.geometry()
            combined_geometry = combined_geometry.union(collection_geometry)
        point_bounds = combined_geometry.bounds()

        for band_id in header_fields:
            # Select the band from the all_bands.
            single_band_image = all_bands.select(band_id)
            band_and_bounds_by_id[f'{band_id}_{year}'] = (
                single_band_image, point_bounds)

    executor.shutdown()

    sample_list = [
        feature
        for future in sample_future_list
        for feature_tuple in zip(*future.result())
        for feature in feature_tuple
        ]
    return header_fields, sample_list, band_and_bounds_by_id


def _process_sample_regions(all_bands, year_points, sample_scale):
    samples = all_bands.reduceRegions(**{
        'collection': year_points,
        'reducer': REDUCER,
        'scale': sample_scale,
    }).getInfo()

    samples_stdev = all_bands.reduceRegions(**{
        'collection': year_points,
        'reducer': ee.Reducer.stdDev(),
        'scale': sample_scale,
    }).getInfo()
    return (samples['features'], samples_stdev['features'])


def get_download_url(global_task_id, task_id, raster_id):
    try:
        LOGGER.debug(f"downloading {global_task_id} - {raster_id}")
        single_band_image, bounds = (
            LOCAL_CONTEXT[global_task_id]['band_and_bounds_by_id'][raster_id])
        url = single_band_image.getDownloadURL({
            'scale': single_band_image.projection().nominalScale(),
            'region': bounds,
            'crs': 'EPSG:4326',
        })
        TASK_LOOKUP[task_id].update({
            'state': 'SUCCESS',
            'result': url})
    except Exception as e:
        LOGGER.exception('something bad happened on get_download_url')
        error_type = type(e).__name__
        error_message = str(e)
        return make_response(jsonify(
            error={'error': f'{error_type}: {error_message}'}), 500)


def get_datasets():
    args_datasets = {}
    ini_path_list = list(glob.glob(os.path.join(INI_DIR, '*.ini')))
    dataset_config_future_list = []
    with ThreadPoolExecutor(len(ini_path_list)) as executor:
        for ini_path in ini_path_list:
            future = executor.submit(parse_ini, ini_path)
            dataset_config_future_list.append((ini_path, future))

    for ini_path, dataset_future in dataset_config_future_list:
        dataset_config = dataset_future.result()
        if dataset_config is None:
            continue
        basename = os.path.basename(os.path.splitext(ini_path)[0])
        args_datasets[basename] = dataset_config
    return args_datasets


def parse_ini(ini_path):
    """Parse ini and return a validated config."""
    try:
        LOGGER.debug(f'parsing {ini_path}')
        basename = os.path.splitext(os.path.basename(ini_path))[0]
        dataset_config = configparser.ConfigParser(allow_no_value=True)
        dataset_config.read(ini_path)
        dataset_result = {}
        if basename not in dataset_config:
            raise ValueError(
                f'expected a section called {basename} but only found '
                f'{dataset_config.sections()}')
        if 'disabled' in dataset_config[basename]:
            LOGGER.debug(f'disabled is in {ini_path}')
            if dataset_config[basename]['disabled'].lower() == 'true':
                LOGGER.debug(f'skipping {ini_path} because disabled')
                return None
        for element_id in EXPECTED_INI_ELEMENTS:
            if element_id not in dataset_config[basename]:
                raise ValueError(
                    f'expected an entry called {element_id} but only found '
                    f'{dataset_config[basename].items()}')
            dataset_result[element_id] = dataset_config[basename][element_id]
        for element_id in OPTIONAL_INI_ELEMENTS:
            if element_id in dataset_config[basename]:
                dataset_result[element_id] = dataset_config[basename][element_id]
        found_expected_section = False
        for section_id in EXPECTED_INI_SECTIONS:
            if section_id in dataset_config:
                found_expected_section = True
                break
        if not found_expected_section:
            raise ValueError(
                f'expected any one of sections called {EXPECTED_INI_SECTIONS} '
                f'but only found {dataset_config.sections()}')
        dataset_result[section_id] = {}
        for element_id in dataset_config[section_id].items():
            element_val = element_id[1]
            try:
                element_val = eval(element_val)
            except NameError:
                pass
            dataset_result[section_id][element_id[0]] = element_val
        dataset_result['bounds'] = eval(dataset_result['bounds'])
        return dataset_result
    except Exception:
        LOGGER.exception(f'error when parsing {ini_path}')
        raise


if __name__ == '__main__':
    app = create_app()
    app.run()
