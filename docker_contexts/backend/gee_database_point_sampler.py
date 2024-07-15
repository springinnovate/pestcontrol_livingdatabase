"""Sample GEE datasets given pest control CSV."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from threading import Lock
import argparse
import bisect
import collections
import logging
import os
import re
import sys

import ee
import pandas
import numpy
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

LIBS_TO_SILENCE = ['urllib3.connectionpool', 'googleapiclient.discovery', 'google_auth_httplib2']
for lib_name in LIBS_TO_SILENCE:
    logging.getLogger(lib_name).setLevel(logging.WARN)

MAX_WORKERS = 10
BATCH_SIZE = 100
DEFAULT_SCALE = 30
MAX_ATTEMPTS = 5
DATATABLE_TEMPLATE_PATH = 'template_datatable.csv'
POINTTABLE_TEMPLATE_PATH = 'template_pointtable.csv'

IGNORE_FLAGS = ['na', 'n/a', 'none']
DATASET_ID = 'Dataset ID'
BAND_NAME = 'Band Name'
SP_TM_AGG_FUNC = 'Spatiotemporal Aggregation Function'
TRANSFORM_FUNC = 'Pixel Value Transform'

SP_TM_AGG_OP = '_internal_sptmaggop'
PIXEL_FN_OP = '_internal_pixelop'

YEARS_FN = 'years'
SPATIAL_FN = 'spatial'
JULIAN_FN = 'julian'

MEAN_STAT = 'mean'
MAX_STAT = 'max'
MIN_STAT = 'min'
MEAN_N_MIN_STAT = 'mean_n_min'
MEAN_N_MAX_STAT = 'mean_n_max'
SD_STAT = 'sd'

OUTPUT_TAG = 'output'
UNIQUE_ID = 'unique_id'

SPATIOTEMPORAL_FN_GRAMMAR = Grammar(r"""
    function = text "(" args (";" function)? ")"
    args = int ("," int)*
    text = ~"[A-Z_]+"i
    int = ~"(\+|-)?[0-9]*"
    """)

MASK_FN = 'mask'
MULT_FN = 'mult'
ADD_FN = 'add'
PIXEL_TRANSFORM_ALLOWED_FUNCTIONS = [
    MASK_FN,
    MULT_FN,
    ADD_FN,
]


ALLOWED_SPATIOTEMPORAL_FUNCTIONS = [
    YEARS_FN,
    JULIAN_FN,
    SPATIAL_FN,
]

N_LIMIT_OPS = [
    MEAN_N_MIN_STAT,
    MEAN_N_MAX_STAT,
]


def initalize_global_stat_functions():
    global VALID_FUNCTIONS
    global IMG_COL_AGGREGATION_FUNCTIONS
    global POINT_AGGREGATION_FUNCTIONS
    global FEATURE_COLLECTION_AGGREGATION_FUNCTIONS
    VALID_FUNCTIONS = [
        f'{prefix}_{suffix}' for suffix in
        [MEAN_STAT, MAX_STAT, MIN_STAT, SD_STAT, MEAN_N_MIN_STAT, MEAN_N_MAX_STAT]
        for prefix in
        [SPATIAL_FN, YEARS_FN, JULIAN_FN]]

    IMG_COL_AGGREGATION_FUNCTIONS = {
        MEAN_STAT: lambda img_col: img_col.mean(),
        MIN_STAT: lambda img_col: img_col.min(),
        MAX_STAT: lambda img_col: img_col.max(),
        SD_STAT: lambda img_col: img_col.reduce(ee.Reducer.stdDev()),
    }

    POINT_AGGREGATION_FUNCTIONS = {
        MEAN_STAT: ee.Reducer.mean().setOutputs([OUTPUT_TAG]),
        MAX_STAT: ee.Reducer.max().setOutputs([OUTPUT_TAG]),
        MIN_STAT: ee.Reducer.min().setOutputs([OUTPUT_TAG]),
        SD_STAT: ee.Reducer.stdDev().setOutputs([OUTPUT_TAG]),
    }

    FEATURE_COLLECTION_AGGREGATION_FUNCTIONS = {
        MEAN_STAT: lambda feature_collection: feature_collection.aggregate_mean(
            OUTPUT_TAG),
        MAX_STAT: lambda feature_collection: feature_collection.aggregate_max(
            OUTPUT_TAG),
        MIN_STAT: lambda feature_collection: feature_collection.aggregate_min(
            OUTPUT_TAG),
        SD_STAT: lambda feature_collection: feature_collection.aggregate_total_sd(
            OUTPUT_TAG),
        MEAN_N_MAX_STAT: lambda feature_collection, n: feature_collection.sort(
            OUTPUT_TAG, False).limit(n).aggregate_mean(OUTPUT_TAG),
        MEAN_N_MIN_STAT: lambda feature_collection, n: feature_collection.sort(
            OUTPUT_TAG, True).limit(n).aggregate_mean(OUTPUT_TAG),
    }


class SpatioTemporalFunctionProcessor(NodeVisitor):
    def __init__(self):
        super()

        self.order_of_ops = []
        self.parsed = collections.defaultdict(bool)

    def visit_function(self, node, visited_children):
        # Extract the function name and arguments
        function_name, _, args, child_fn = visited_children[0:4]
        if function_name not in VALID_FUNCTIONS:
            raise ValueError(
                f'unexpected function: "{function_name}" '
                f'must be one of {VALID_FUNCTIONS}')
        #LOGGER.debug(f'execute {function_name} with {args}')
        function, stat_operation = function_name.split('_', 1)
        if self.parsed[function]:
            raise ValueError(
                f'{function} already seen once, cannot apply it twice')
        if function == JULIAN_FN and self.parsed[YEARS_FN]:
            raise ValueError(
                f'{node.text} cannot apply a julian aggregation after a year aggregation')
        if function in [JULIAN_FN, YEARS_FN]:
            if (stat_operation not in [MEAN_N_MIN_STAT, MEAN_N_MAX_STAT]) and (len(args) != 2):
                raise ValueError(
                    f'in `{node.text}`, `{function}` requires two arguments, got '
                    f'this instead: {args}')
            elif (stat_operation in [MEAN_N_MIN_STAT, MEAN_N_MAX_STAT]) and (len(args) != 3):
                raise ValueError(
                    f'in `{node.text}`, `{function}` requires three arguments, got '
                    f'this instead: {args}')
        if function in [SPATIAL_FN] and len(args) != 1:
            raise ValueError(
                f'in `{node.text}`, `{function}` requires one argument, got '
                f'this instead: {args}')
        self.parsed[function] = True
        if isinstance(child_fn, list):
            function_chain = child_fn[0][1]
            function_chain.append((function, stat_operation, args))
            return function_chain
        else:
            return [(function, stat_operation, args)]

    def visit_args(self, node, visited_children):
        # Process and collect arguments
        first_int = visited_children[0]
        integers = [first_int]
        # get rid of the commas
        for possible_arg in visited_children[1]:
            if len(possible_arg) == 2:
                # comma and number
                integers.append(possible_arg[1])
        return integers

    def visit_text(self, node, visited_children):
        # Return the text directly
        return node.text

    def visit_int(self, node, visited_children):
        # Convert and return the integer
        return int(node.text)

    def generic_visit(self, node, visited_children):
        return visited_children or node


DESCRIPTION_FIELD = 'Description'
EXPECTED_DATATABLE_COLUMNS = [
    BAND_NAME,
    SP_TM_AGG_FUNC,
    TRANSFORM_FUNC,
    DATASET_ID,
]

LAT_FIELD = 'Latitude'
LNG_FIELD = 'Longitude'
YEAR_FIELD = 'Year'

EXPECTED_POINTTABLE_COLUMNS = [
    LAT_FIELD,
    LNG_FIELD,
    YEAR_FIELD
]


def process_batch(batch_features, active_collection, batch_size, op_type, local_scale):
    def reduce_image(image):
        reduced_points = image.reduceRegions(
            collection=batch_features,
            reducer=POINT_AGGREGATION_FUNCTIONS[op_type],
            scale=local_scale,
        )
        time_start_millis = image.get('system:time_start')
        return reduced_points.map(lambda feature: feature.set('system:time_start', time_start_millis))

    batch_results = active_collection.map(reduce_image).flatten()
    # Check if 'output' property is present in the first feature of the batch results
    first_feature = ee.Feature(batch_results.first())
    #output_present = first_feature.propertyNames().contains('output').getInfo()
    output_present = True
    return (batch_results, output_present)


def _scrub_key(key):
    key = key.replace(' ', '')
    key = re.sub(r'[,.());\\/]', '_', key)
    key = re.sub(r'_+', '_', key)
    key = key.replace('_nan', '')
    return key


# Define a function to expand year ranges into individual years
def expand_year_range(year_field):
    def _expand_year_range(row):
        if '-' in str(row[year_field]):
            start_year, end_year = map(int, row[year_field].split('-'))
            return [{**row, year_field: str(year)} for year in range(start_year, end_year + 1)]

        else:
            return [{**row}]
    return _expand_year_range


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


def chunk_points(point_list, chunk_size):
    """Yield successive chunks from point_list."""
    for i in range(0, len(point_list), chunk_size):
        yield point_list[i:i + chunk_size]


def initalize_gee(authenicate_flag):
    if authenicate_flag:
        ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    initalize_global_stat_functions()


def create_clean_key(dataset_row):
    key = '_'.join(
        str(dataset_row[col_id]) for col_id in EXPECTED_DATATABLE_COLUMNS
        if isinstance(dataset_row[col_id], str) or not numpy.isnan(dataset_row[col_id]))
    if DESCRIPTION_FIELD in dataset_row:
        description_val = dataset_row[DESCRIPTION_FIELD]
        if description_val not in [None, '']:
            key = f'{description_val} {key}'
    # Define the regular expression to match all punctuation except '_'
    punctuation_regex = r'[^\w]'

    # Use re.sub() to replace matched characters with an empty string
    key = re.sub(punctuation_regex, '_', key)
    key = re.sub(r'__+', '_', key)
    return key


def process_data_table(dataset_table, data_table_attributes):
    missing_columns = set(
        EXPECTED_DATATABLE_COLUMNS).difference(set(dataset_table.columns))
    if missing_columns:
        raise ValueError(
            'expected the following columns in the data table that were ' +
            'missing:\n' + '\n\t'.join(missing_columns) +
            '\nexisting columns:' + '\n\t'.join(dataset_table.columns))

    for row_index, dataset_row in dataset_table.iterrows():
        key = create_clean_key(dataset_row)
        if isinstance(dataset_row[TRANSFORM_FUNC], str) and \
                dataset_row[TRANSFORM_FUNC].lower() not in IGNORE_FLAGS:
            transform_func_re = re.search(
                r'([^\[]*)(\[.*\])', dataset_row[TRANSFORM_FUNC])
            if transform_func_re is None:
                raise ValueError(
                    f'"{dataset_row[TRANSFORM_FUNC]}" doesn\'t match any '
                    'Pixel Value Transform function')
            pixel_fn, pixel_args = re.search(
                r'([^\[]*)(\[.*\])', dataset_row[TRANSFORM_FUNC]).groups()
            if pixel_fn not in PIXEL_TRANSFORM_ALLOWED_FUNCTIONS:
                raise NotImplementedError(
                    f'{dataset_row[TRANSFORM_FUNC]} is not a valid pixel '
                    'transform, choose one of these '
                    '{PIXEL_TRANSFORM_ALLOWED_FUNCTIONS}')
            dataset_table.at[row_index, PIXEL_FN_OP] = [
                pixel_fn, eval(pixel_args)]
        spatiotemporal_fn = dataset_row[SP_TM_AGG_FUNC]
        spatiotemporal_fn = spatiotemporal_fn.replace(' ', '')
        try:
            grammar_tree = SPATIOTEMPORAL_FN_GRAMMAR.parse(spatiotemporal_fn)
        except:
            raise ValueError(
                f'the function "{spatiotemporal_fn}" could not be parsed, check '
                f'for syntax errors\n error on row {dataset_row}')
        lexer = SpatioTemporalFunctionProcessor()
        output = lexer.visit(grammar_tree)
        dataset_table.at[row_index, SP_TM_AGG_OP] = output
    try:
        dataset_table.to_csv('test.csv')
    except PermissionError:
        LOGGER.warning(f'test.csv is open, cannot overwrite')
    LOGGER.info(f'loaded {len(dataset_table)} datasets')
    return dataset_table


def generate_templates():
    for template_path, columns in [
            (DATATABLE_TEMPLATE_PATH, EXPECTED_DATATABLE_COLUMNS),
            (POINTTABLE_TEMPLATE_PATH, EXPECTED_POINTTABLE_COLUMNS)]:
        if os.path.exists(template_path):
            LOGGER.warning(
                f'{template_path} already exists, not overwriting')
            continue
        with open(template_path, 'w') as datatable:
            datatable.write(','.join(columns))
        LOGGER.info(f'wrote template to {template_path}')


INFER_RESOLUTION_MEMO_CACHE = {}
INFER_RESOLUTION_MEMO_LOCK = Lock()


def infer_temporal_and_spatial_resolution_and_valid_years(collection):
    with INFER_RESOLUTION_MEMO_LOCK:
        if collection not in INFER_RESOLUTION_MEMO_CACHE:
            date_collection = collection.aggregate_array('system:time_start')
            year_strings = date_collection.map(
                lambda timestamp: ee.Date(timestamp).format('YYYY'))
            unique_years = set(
                int(x) for x in ee.List(year_strings).distinct().getInfo())
            date_list = [
                datetime.utcfromtimestamp(date/1000)
                for date in date_collection.getInfo()]
            scale = collection.first().projection().nominalScale().getInfo()
            for i in range(1, len(date_list)):
                diff = (date_list[i] - date_list[i-1]).days
                if diff > 300:  # we saw at least a year's gap, so quit
                    return YEARS_FN, scale, unique_years
            INFER_RESOLUTION_MEMO_CACHE[collection] = (
                JULIAN_FN, scale, unique_years)
        return INFER_RESOLUTION_MEMO_CACHE[collection]


def get_year_julian_range(current_year, spatiotemporal_commands):
    """Returns offset of year in [min, max+1) range"""
    # Initialize current_year as the first day of this year
    year_range = (current_year, current_year+1)
    julian_range = (1, 365)
    for spatiotemp_flag, op_type, args in spatiotemporal_commands:
        if spatiotemp_flag == YEARS_FN:
            year_range = (
                year_range[0]+args[0],
                year_range[1]+args[1])
        if spatiotemp_flag == JULIAN_FN:
            if len(args) == 2:
                # start and end date
                julian_range = tuple(args)
            else:
                # startss with an 'n'
                julian_range = tuple(args[1:3])
    return year_range, julian_range


def filter_imagecollection_by_date_range(year_range, julian_range, image_collection):
    # Initialize an empty ImageCollection
    filtered_collection = ee.ImageCollection([])

    for year in range(year_range[0], year_range[1]):
        if julian_range[0] < 0:
            # For -1 to mean December 31 of the previous year, add 1 less than
            # the Julian day because -1 should add 0 days to December 31
            start_date = date(year - 1, 12, 31) + timedelta(days=julian_range[0] + 1)
        else:
            # Julian day 1 should be January 1st, so subtract 1 from the Julian day
            start_date = date(year, 1, 1) + timedelta(days=julian_range[0] - 1)

        if julian_range[1] > 365:
            # Subtract 365 to find the overflow and subtract an additional 1 since
            # January 1st should represent the 1st Julian day of the next year
            # but enddate is exclusive so set it off by 1 day
            end_date = date(year + 1, 1, 1) + timedelta(days=julian_range[1] - 365)
        else:
            # If within the same year, just adjust from January 1st
            end_date = date(year, 1, 1) + timedelta(days=julian_range[1])

        # Convert dates to strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Filter the image collection for the given range and merge with the cumulative collection
        year_filtered_collection = image_collection.filterDate(start_date_str, end_date_str)
        filtered_collection = filtered_collection.merge(year_filtered_collection)
    return filtered_collection


LOAD_DATASET_MEMO_CACHE = {}
LOAD_DATASET_MEMO_LOCK = Lock()


def load_collection_or_image(dataset_id):
    # Check if the result is already in the cache
    with LOAD_DATASET_MEMO_LOCK:
        if dataset_id not in LOAD_DATASET_MEMO_CACHE:
            dataset = ee.ImageCollection(dataset_id)
            try:
                # Try to access a property that should be present in an ImageCollection
                # If the property access succeeds, it's likely an ImageCollection
                dataset.size().getInfo()
                LOAD_DATASET_MEMO_CACHE[dataset_id] = dataset
                return dataset
            except ee.EEException:
                # If an error occurred, it's likely not an ImageCollection, so load as an Image
                image = ee.Image(dataset_id)
                # Convert the single Image to an ImageCollection
                image_collection = ee.ImageCollection([image])
                LOAD_DATASET_MEMO_CACHE[dataset_id] = image_collection
        return LOAD_DATASET_MEMO_CACHE[dataset_id]


def process_gee_dataset(
        dataset_id,
        band_name,
        point_list_by_year,
        point_unique_id_per_year,
        pixel_op_transform,
        spatiotemporal_commands,
        target_point_table_path):
    """Apply the commands in the `commands` list to generate the appropriate result"""
    # make sure that the final opeation is a spatial one if not alreay defined
    LOGGER.info(
        f'processing the following commands:\n'
        f'\t{dataset_id} - {band_name} -- {spatiotemporal_commands}')
    if SPATIAL_FN not in [x[0] for x in spatiotemporal_commands]:
        spatiotemporal_commands += [(SPATIAL_FN, MEAN_STAT, [0])]

    image_collection = load_collection_or_image(dataset_id)
    image_collection = image_collection.select(band_name)
    collection_temporal_resolution, nominal_scale, valid_year_set = \
        infer_temporal_and_spatial_resolution_and_valid_years(
            image_collection)

    collection_per_year = ee.Dictionary()
    for current_year in point_list_by_year.keys():
        applied_functions = set()
        # apply year filter
        year_range, julian_range = get_year_julian_range(
            current_year, spatiotemporal_commands)

        if valid_year_set and any(
                year not in valid_year_set for year in year_range):
            LOGGER.debug(
                f'{valid_year_set} does not cover queried year range '
                f'of {year_range} in {dataset_id} - {band_name}')
            collection_per_year = collection_per_year.set(str(current_year), ee.List([
                (unique_id,
                 f'{valid_year_set} does not cover queried year range of {year_range}')
                for unique_id in point_unique_id_per_year[current_year]]))
            continue
        if valid_year_set:
            active_collection = filter_imagecollection_by_date_range(
                year_range, julian_range, image_collection)
        else:
            active_collection = image_collection
        if pixel_op_transform is not None:
            def pixel_op(pixel_op_fn, args):
                return {
                    MASK_FN: lambda: active_collection.map(
                        lambda image: image.remap(
                            ee.List(args),
                            ee.List([1]*len(args)), 0).copyProperties(
                                image, ['system:time_start'])),
                    MULT_FN: lambda: active_collection.map(
                        lambda image: image.multiply(
                            ee.Image(args)).copyProperties(
                                image, ['system:time_start'])),
                    ADD_FN: lambda: active_collection.map(
                        lambda image: image.add(
                            ee.Image(args)).copyProperties(
                                image, ['system:time_start'])),
                }.get(pixel_op_fn, lambda: None)()

            pixel_op_fn, pixel_op_args = pixel_op_transform
            active_collection = pixel_op(pixel_op_fn, pixel_op_args)
            if active_collection is None:
                raise ValueError(
                    f'"{pixel_op_fn}" is not a valid function in '
                    f'{PIXEL_TRANSFORM_ALLOWED_FUNCTIONS} for {dataset_id} - {band_name}')
        n_points = len(point_list_by_year[current_year])
        LOGGER.info(f'processing {n_points} points on {dataset_id} {band_name} {pixel_op_transform} {spatiotemporal_commands} {current_year} over {year_range} on {os.path.basename(target_point_table_path)}')
        for index, (spatiotemp_flag, op_type, args) in enumerate(spatiotemporal_commands):
            point_list = ee.FeatureCollection(point_list_by_year[current_year])
            if spatiotemp_flag in applied_functions:
                raise ValueError(
                    f'already applied a {spatiotemp_flag} in the command list '
                    f'{spatiotemporal_commands}')
            applied_functions.add(spatiotemp_flag)
            if (spatiotemp_flag == JULIAN_FN and
                    collection_temporal_resolution == YEARS_FN):
                raise ValueError(
                    f'requesting {spatiotemp_flag} when underlying '
                    f'dataset is coarser at {collection_temporal_resolution} '
                    f'for {dataset_id} - {band_name}')

            # process the collection on this spatiotemporal function
            if isinstance(active_collection, ee.ImageCollection):
                if spatiotemp_flag == YEARS_FN:
                    # already been filtered to be the right year span, just
                    # set the target time to be the current year
                    time_start_millis = ee.Date.fromYMD(
                        current_year, 1, 1).millis()
                    active_collection = ee.ImageCollection(
                        IMG_COL_AGGREGATION_FUNCTIONS[op_type](
                            active_collection).set(
                            'system:time_start', time_start_millis))
                elif spatiotemp_flag == JULIAN_FN:
                    if op_type in N_LIMIT_OPS:
                        raise RuntimeError(f'{spatiotemp_flag} is not implemented for images')
                    start_day, end_day = args

                    def _op_by_julian_range(_year):
                        # aggregate julian range around _year to just
                        # a single image at _year
                        start_date = ee.Date.fromYMD(_year, 1, 1)
                        if start_day > 0:
                            # day 1 should be jan 1, so do a -1
                            start_date = start_date.advance(start_day-1, 'day')
                        else:
                            start_date = start_date.advance(start_day, 'day')
                        end_date = ee.Date.fromYMD(_year, 1, 1)
                        if end_day <= 365:
                            end_date = end_date.advance(end_day, 'day')
                        else:
                            end_date = end_date.advance(end_day-1, 'day')
                        daily_collection = active_collection.filterDate(
                            start_date, end_date)
                        time_start_millis = ee.Date.fromYMD(
                            _year, 1, 1).millis()
                        aggregate_image = IMG_COL_AGGREGATION_FUNCTIONS[
                            op_type](daily_collection)
                        aggregate_image = aggregate_image.set(
                                'system:time_start', time_start_millis)
                        return aggregate_image
                    # Defines as the min/max year that will be aggregated
                    # later for current year
                    years = ee.List(list(range(*year_range)))
                    active_collection = ee.ImageCollection.fromImages(
                        years.map(lambda y: _op_by_julian_range(y)))
                elif spatiotemp_flag == SPATIAL_FN:
                    if args[0] > 0:
                        buffered_point_list = point_list.map(
                            lambda feature: feature.buffer(args[0]))
                    else:
                        buffered_point_list = ee.FeatureCollection(point_list)

                    local_scale = (
                        nominal_scale if nominal_scale < args[0] and args[0] > 0
                        else DEFAULT_SCALE)

                    batch_size = BATCH_SIZE
                    i = 0
                    results = ee.FeatureCollection([])
                    while i < n_points:
                        batch = buffered_point_list.toList(batch_size, i)
                        batch_features = ee.FeatureCollection(batch)
                        batch_results, output_present = process_batch(
                            batch_features, active_collection, batch_size,
                            op_type, local_scale)
                        if output_present:
                            # If 'output' is present, proceed with the next batch
                            results = results.merge(batch_results)
                            i += batch_size
                        else:
                            # If 'output' is missing, halve the batch size and retry
                            if batch_size > 1:
                                batch_size = max(1, batch_size // 2)
                            else:
                                raise RuntimeError(
                                    "Minimum batch size reached with "
                                    "missing 'output' property.")
                    active_collection = results
            elif isinstance(active_collection, ee.FeatureCollection):
                if spatiotemp_flag == YEARS_FN:
                    # we group all the points into a single value for the year?
                    def reduce_by_unique_id(unique_id):
                        unique_collection = active_collection.filter(
                            ee.Filter.eq(UNIQUE_ID, unique_id))
                        aggregate_output = \
                            FEATURE_COLLECTION_AGGREGATION_FUNCTIONS[op_type](
                                unique_collection)
                        representative_feature = ee.Feature(
                            unique_collection.first()).set(
                                OUTPUT_TAG, aggregate_output)
                        return representative_feature
                    # Get a list of unique IDs and years to iterate over.
                    active_collection = ee.FeatureCollection(
                        ee.List(point_unique_id_per_year[current_year]).map(
                            lambda unique_id: reduce_by_unique_id(
                                unique_id)))
                elif spatiotemp_flag == JULIAN_FN:
                    # we group all the points into groups of years
                    # Function to calculate mean output by unique_id and year.
                    if op_type in N_LIMIT_OPS:
                        n_samples, start_day, end_day = args
                    else:
                        start_day, end_day = args

                    def reduce_by_julian(unique_id, _year):
                        start_date = ee.Date.fromYMD(_year, 1, 1)
                        if args[0] > 0:
                            # day 1 should be jan 1, so do a -1
                            start_date = start_date.advance(start_day-1, 'day')
                        else:
                            start_date = start_date.advance(start_day, 'day')
                        end_date = ee.Date.fromYMD(_year, 1, 1)
                        if end_day < 365:
                            end_date = end_date.advance(end_day-1, 'day')
                        else:
                            end_date = end_date.advance(end_day, 'day')

                        julian_collection = active_collection.filterDate(
                            start_date, end_date).filter(ee.Filter.eq(
                                UNIQUE_ID, unique_id))

                        if op_type in N_LIMIT_OPS:
                            aggregate_output = FEATURE_COLLECTION_AGGREGATION_FUNCTIONS[
                                op_type](julian_collection, n_samples)
                        else:
                            aggregate_output = FEATURE_COLLECTION_AGGREGATION_FUNCTIONS[
                                op_type](julian_collection)

                        representative_feature = ee.Feature(
                            julian_collection.first()).set(
                                OUTPUT_TAG, aggregate_output)
                        return representative_feature

                    years = ee.List(list(range(*year_range)))
                    # Use nested mapping to apply the mean function to each unique ID and year combination.
                    active_collection = ee.FeatureCollection(
                        ee.List(point_unique_id_per_year[current_year]).map(
                            lambda unique_id: years.map(
                                lambda year: reduce_by_julian(
                                    unique_id, year))).flatten())

        def accumulate_by_year(active_collection):
            def extract_properties(feature):
                return ee.Feature(None, {
                    UNIQUE_ID: feature.get(UNIQUE_ID),
                    OUTPUT_TAG: feature.get(OUTPUT_TAG)
                })

            processed_features = active_collection.map(extract_properties)
            result = processed_features.reduceColumns(
                ee.Reducer.toList(2), [UNIQUE_ID, OUTPUT_TAG]).get('list')
            return result

        try:
            accumulated_active_collection = accumulate_by_year(active_collection)
            collection_per_year = collection_per_year.set(
                str(current_year), accumulated_active_collection)
        except Exception:
            LOGGER.exception(
                f'something bad happened on this collectoin: {active_collection}')
            sys.exit()
            raise

    collection_per_year_info = collection_per_year.getInfo()
    # Convert the dictionary to your desired format
    result_list = []
    for year, data in collection_per_year_info.items():
        try:
            result_list.extend([(f[0], f[1]) for f in data])
        except Exception:
            LOGGER.exception(f'something bad happened on this collection: {data}')
            raise

    return result_list


def _debug_save_image(active_collection, desc):
    half_side_length = 0.1  # Change this to half of your desired side length
    center_lng = -72.5688086775836
    center_lat = 42.43584566068613
    coordinates = [
        [center_lng - half_side_length, center_lat - half_side_length],  # Lower-left corner
        [center_lng + half_side_length, center_lat - half_side_length],  # Lower-right corner
        [center_lng + half_side_length, center_lat + half_side_length],  # Upper-right corner
        [center_lng - half_side_length, center_lat + half_side_length],  # Upper-left corner
        [center_lng - half_side_length, center_lat - half_side_length]   # Closing the loop
    ]

    LOGGER.debug(active_collection.getInfo())
    image = ee.Image(active_collection.first())
    LOGGER.debug(f'scale: {image.projection().nominalScale().getInfo()}')
    LOGGER.debug(image.getInfo())
    square = ee.Geometry.Polygon([coordinates])
    export = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        scale=30,
        region=square)  # Define the region
    export.start()
    LOGGER.debug(export.status())


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='sample points on GEE data')
    parser.add_argument(
        '--generate_templates', action='store_true', help=(
            'Generate template tables and then quit no matter what '
            'other arguments are passed.'))
    parser.add_argument(
        '--authenticate', action='store_true',
        help='Pass this flag if you need to reauthenticate with GEE')
    parser.add_argument(
        '--dataset_table_path', required=True, help='path to data table')
    parser.add_argument(
        '--point_table_path', help='path to point sample locations',
        required=True)
    parser.add_argument(
        '--n_dataset_rows', nargs='+', type=int, help='limit csv read to this many rows')
    parser.add_argument(
        '--n_point_rows', type=int, help='limit csv read to this many rows')
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help=f'how many points to process per batch, defaults to {BATCH_SIZE}')
    parser.add_argument(
        '--max_workers', type=int, default=MAX_WORKERS,
        help=f'how many datasets to process in parallel, defaults to {MAX_WORKERS}')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon

    args = parser.parse_args()
    if args.generate_templates:
        generate_templates()
        return
    initalize_gee(args.authenticate)

    dataset_table = pandas.read_csv(
        args.dataset_table_path,
        skip_blank_lines=True,
    ).dropna(how='all')

    if args.n_dataset_rows is None:
        dataset_row_range = range(0, len(dataset_table))
    elif len(args.n_dataset_rows) == 1:
        dataset_row_range = range(0, args.n_dataset_rows[0])
    elif len(args.n_dataset_rows) == 2:
        dataset_row_range = range(args.n_dataset_rows[0], args.n_dataset_rows[1])

    dataset_table[SP_TM_AGG_OP] = None
    dataset_table[PIXEL_FN_OP] = None
    dataset_table = process_data_table(dataset_table, args)
    LOGGER.info(f'loaded {args.dataset_table_path}')
    point_table = pandas.read_csv(
        args.point_table_path,
        skip_blank_lines=True,
        converters={
            LNG_FIELD: lambda x: None if x == '' else float(x),
            LAT_FIELD: lambda x: None if x == '' else float(x),
        },
        nrows=args.n_point_rows).dropna(how='all')

    # Explode out the YYYY-YYYY column to individual rows
    point_table = pandas.DataFrame(point_table.apply(
        expand_year_range(YEAR_FIELD), axis=1).sum())
    point_table[YEAR_FIELD] = point_table[YEAR_FIELD].astype(int)

    LOGGER.info(f'loaded {args.point_table_path}')
    target_point_table_path = (
        f'{os.path.basename(os.path.splitext(args.dataset_table_path)[0])}_'
        f'{os.path.basename(os.path.splitext(args.point_table_path)[0])}')

    point_batch_list = []
    point_unique_id_per_year_list = []
    n_points = 0
    # initialize for first step
    batch_index = -1
    current_batch_size = args.batch_size
    for index, (year, lon, lat) in enumerate(zip(
            point_table[YEAR_FIELD],
            point_table[LNG_FIELD],
            point_table[LAT_FIELD])):
        if current_batch_size == args.batch_size:
            batch_index += 1
            point_batch_list.append(collections.defaultdict(list))
            point_unique_id_per_year_list.append(collections.defaultdict(list))
            current_batch_size = 0

        point_batch_list[batch_index][year].append(
            ee.Feature(ee.Geometry.Point(
                [lon, lat], 'EPSG:4326'), {UNIQUE_ID: index}))
        point_unique_id_per_year_list[batch_index][year].append(index)
        n_points += 1
        current_batch_size += 1

    def process_dataset_row(dataset_index, dataset_row):
        key = (
            f'{dataset_row[DATASET_ID]}_'
            f'{dataset_row[BAND_NAME]}_'
            f'{dataset_row[SP_TM_AGG_FUNC]}_'
            f'{dataset_row[TRANSFORM_FUNC]}')
        key = _scrub_key(key)
        result_by_index = {}
        for batch_index, (point_features_by_year, point_unique_id_per_year) in enumerate(zip(point_batch_list, point_unique_id_per_year_list)):
            LOGGER.debug(f'BATCH INDEX {batch_index}')
            n_attempts = 0
            while True:
                try:
                    point_collection_by_year = process_gee_dataset(
                        dataset_row[DATASET_ID],
                        dataset_row[BAND_NAME],
                        point_features_by_year,
                        point_unique_id_per_year,
                        dataset_row[PIXEL_FN_OP],
                        dataset_row[SP_TM_AGG_OP],
                        target_point_table_path)
                    break
                except Exception:
                    if n_attempts == MAX_ATTEMPTS:
                        LOGGER.exception('ERROR ON process_gee_dataset')
                        raise
                    else:
                        n_attempts += 1
                        LOGGER.exception(f'trying attempt number {n_attempts+1} of {MAX_ATTEMPTS} ON {key}')
            LOGGER.debug(f'completed BATCH {batch_index+1} of {len(point_batch_list)}')
            for point_index, point_result in point_collection_by_year:
                result_by_index[point_index] = point_result
        result_by_order = [
            'n/a' if index not in result_by_index
            else result_by_index[index] for index in range(n_points)]

        return key, result_by_order

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submitting tasks to the executor
        futures = {
            executor.submit(
                process_dataset_row, dataset_index, dataset_row): dataset_index
            for dataset_index, dataset_row in dataset_table.iterrows()
            if dataset_index in dataset_row_range}

        # Retrieving results as they complete
        for future in futures:
            dataset_index = futures[future]
            try:
                key, result_by_order = future.result()
                point_table[key] = result_by_order
                LOGGER.info(f'{key} just finished!!')
            except Exception as exc:
                LOGGER.exception(f'{dataset_index} generated an exception: {exc}')
                point_table[key] = [str(exc)] * n_points

    point_table.to_csv(
        f'sampled_points_of_{target_point_table_path}.csv', index=False)
    LOGGER.info(f'all done to {target_point_table_path}!')


if __name__ == '__main__':
    main()
