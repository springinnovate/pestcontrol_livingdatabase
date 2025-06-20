"""Sample GEE datasets given pest control CSV."""

from datetime import datetime, timedelta, date
from threading import Lock
import argparse
import bisect
import collections
import logging
import os
import re
import sys
import time

import requests
from database import DATABASE_URI
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import ee
import pandas

engine = create_engine(DATABASE_URI, echo=False, connect_args={"timeout": 30})
Session = scoped_session(sessionmaker(bind=engine))
session_factory = sessionmaker(bind=engine)


DB_COMMIT_LOCK = Lock()

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
logging.getLogger("taskgraph").setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

LIBS_TO_SILENCE = [
    "urllib3.connectionpool",
    "googleapiclient.discovery",
    "google_auth_httplib2",
]
for lib_name in LIBS_TO_SILENCE:
    logging.getLogger(lib_name).setLevel(logging.WARN)

MAX_WORKERS = 10
BATCH_SIZE = 100
N_POINTS_BATCH = 50
DEFAULT_SCALE = 30
MAX_ATTEMPTS = 5
DATATABLE_TEMPLATE_PATH = "template_datatable.csv"
POINTTABLE_TEMPLATE_PATH = "template_pointtable.csv"

IGNORE_FLAGS = ["na", "n/a", "none"]
DATASET_ID = "Dataset ID"
BAND_NAME = "Band Name"
SP_TM_AGG_FUNC = "Spatiotemporal Aggregation Function"
TRANSFORM_FUNC = "Pixel Value Transform"
START_DATE = "start_date"
END_DATE = "end_date"
DATASET_TYPE = "dataset_type"
COLLECTION_TEMPORAL_RESOLUTION = "collection_temporal_resolution"
NOMINAL_SCALE = "nominal_scale"

SP_TM_AGG_OP = "_internal_sptmaggop"
PIXEL_FN_OP = "_internal_pixelop"

YEARS_FN = "years"
SPATIAL_FN = "spatial"
JULIAN_FN = "julian"

MEAN_STAT = "mean"
MAX_STAT = "max"
MIN_STAT = "min"
MEAN_N_MIN_STAT = "mean_n_min"
MEAN_N_MAX_STAT = "mean_n_max"
SD_STAT = "sd"

OUTPUT_TAG = "output"
UNIQUE_ID = "unique_id"

SPATIOTEMPORAL_FN_GRAMMAR = Grammar(
    r"""
    function = text "(" args (";" function)? ")"
    args = int ("," int)*
    text = ~"[A-Z_]+"i
    int = ~"(\+|-)?[0-9]*"
    """
)

MASK_FN = "mask"
MULT_FN = "mult"
ADD_FN = "add"
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

DESCRIPTION_FIELD = "Description"
EXPECTED_DATATABLE_COLUMNS = [
    BAND_NAME,
    SP_TM_AGG_FUNC,
    TRANSFORM_FUNC,
    DATASET_ID,
]

LAT_FIELD = "Latitude"
LNG_FIELD = "Longitude"
YEAR_FIELD = "Year"

EXPECTED_POINTTABLE_COLUMNS = [LAT_FIELD, LNG_FIELD, YEAR_FIELD]


def parse_spatiotemporal_fn(spatiotemporal_fn):
    """Parse the `spatiotemporal_fn` as a string into the function grammar."""
    spatiotemporal_fn = spatiotemporal_fn.replace(" ", "")
    grammar_tree = SPATIOTEMPORAL_FN_GRAMMAR.parse(spatiotemporal_fn)
    lexer = SpatioTemporalFunctionProcessor()
    output = lexer.visit(grammar_tree)
    return output


def parse_gee_dataset_info(url):
    """Read the GEE developer page for dataset range, ID, and resolution."""
    if not url.startswith("https://"):
        base_dataset_id = url.replace("/", "_")
        url = f"https://developers.google.com/earth-engine/datasets/catalog/{base_dataset_id}"
    result = collections.defaultdict(lambda: None)
    r = requests.get(url)

    # Parse dataset availability dates
    date_match = re.search(
        r"Dataset Availability.*?(\d{4}-\d{2}-\d{2})T.*?(\d{4}-\d{2}-\d{2})T",
        r.text,
        re.DOTALL,
    )
    if date_match:
        start_date, end_date = date_match.groups()
        result[START_DATE], result[END_DATE] = start_date, end_date

    # Parse Earth Engine Snippet
    dataset_match = re.search(
        r"Earth Engine Snippet.*?(ee\.(ImageCollection|Image)\(\"([^\"]+)\"\))",
        r.text,
        re.DOTALL,
    )
    if dataset_match:
        _, result[DATASET_TYPE], result[DATASET_ID] = dataset_match.groups()

    # Parse temporal resolution (cadence)
    temporal_match = re.search(
        r"Cadence.*?(\bYear\b|\bMonth\b|\bDay\b)", r.text, re.DOTALL
    )
    if temporal_match:
        cadence = temporal_match.group(1)
        if cadence == "Year":
            result[COLLECTION_TEMPORAL_RESOLUTION] = YEARS_FN
        else:
            result[COLLECTION_TEMPORAL_RESOLUTION] = JULIAN_FN

    # Parse nominal scale (resolution)
    resolution_match = re.search(
        r"<p>\s*<b>Resolution</b>\s*<br>\s*([\d,]+)\s*meters", r.text, re.DOTALL
    )
    if resolution_match:
        resolution = resolution_match.group(1).replace(",", "")
        result[NOMINAL_SCALE] = int(resolution)

    resolution_match = re.search(
        r"Pixel Size.*?(\d+) meters", r.text, re.DOTALL
    )
    if resolution_match:
        resolution = resolution_match.group(1).replace(",", "")
        result[NOMINAL_SCALE] = int(resolution)

    return result


def point_table_to_point_batch(csv_file, n_rows=None):
    point_table = pandas.read_csv(
        csv_file,
        dtype={YEAR_FIELD: int, LNG_FIELD: float, LAT_FIELD: float},
        nrows=n_rows,
    )

    point_features_by_year = collections.defaultdict(list)
    point_unique_id_per_year = collections.defaultdict(list)
    points_per_year = collections.defaultdict(int)
    for index, row in point_table.iterrows():
        year = int(row[YEAR_FIELD])
        batch_id = points_per_year[year] // N_POINTS_BATCH
        batch_key = f"{year}_{batch_id}"
        if len(point_features_by_year[batch_key]) >= N_POINTS_BATCH:
            batch_id += 1
            batch_key = f"{year}_{batch_id}"
        point_features_by_year[batch_key].append(
            ee.Feature(
                ee.Geometry.Point(
                    [row[LNG_FIELD], row[LAT_FIELD]], "EPSG:4326"
                ),
                {UNIQUE_ID: index},
            )
        )
        point_unique_id_per_year[batch_key].append(index)
        points_per_year[year] += 1
    return point_features_by_year, point_unique_id_per_year, point_table


def initalize_global_stat_functions():
    global VALID_FUNCTIONS
    global IMG_COL_AGGREGATION_FUNCTIONS
    global POINT_AGGREGATION_FUNCTIONS
    global FEATURE_COLLECTION_AGGREGATION_FUNCTIONS
    VALID_FUNCTIONS = [
        f"{prefix}_{suffix}"
        for suffix in [
            MEAN_STAT,
            MAX_STAT,
            MIN_STAT,
            SD_STAT,
            MEAN_N_MIN_STAT,
            MEAN_N_MAX_STAT,
        ]
        for prefix in [SPATIAL_FN, YEARS_FN, JULIAN_FN]
    ]

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
            OUTPUT_TAG
        ),
        MAX_STAT: lambda feature_collection: feature_collection.aggregate_max(
            OUTPUT_TAG
        ),
        MIN_STAT: lambda feature_collection: feature_collection.aggregate_min(
            OUTPUT_TAG
        ),
        SD_STAT: lambda feature_collection: feature_collection.aggregate_total_sd(
            OUTPUT_TAG
        ),
        MEAN_N_MAX_STAT: lambda feature_collection, n: feature_collection.sort(
            OUTPUT_TAG, False
        )
        .limit(n)
        .aggregate_mean(OUTPUT_TAG),
        MEAN_N_MIN_STAT: lambda feature_collection, n: feature_collection.sort(
            OUTPUT_TAG, True
        )
        .limit(n)
        .aggregate_mean(OUTPUT_TAG),
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
                f"must be one of {VALID_FUNCTIONS}"
            )
        function, stat_operation = function_name.split("_", 1)
        if self.parsed[function]:
            raise ValueError(
                f"{function} already seen once, cannot apply it twice"
            )
        if function == JULIAN_FN and self.parsed[YEARS_FN]:
            raise ValueError(
                f"{node.text} cannot apply a julian aggregation after a year aggregation"
            )
        if function in [JULIAN_FN, YEARS_FN]:
            if (stat_operation not in [MEAN_N_MIN_STAT, MEAN_N_MAX_STAT]) and (
                len(args) != 2
            ):
                raise ValueError(
                    f"in `{node.text}`, `{function}` requires two arguments, got "
                    f"this instead: {args}"
                )
            elif (stat_operation in [MEAN_N_MIN_STAT, MEAN_N_MAX_STAT]) and (
                len(args) != 3
            ):
                raise ValueError(
                    f"in `{node.text}`, `{function}` requires three arguments, got "
                    f"this instead: {args}"
                )
        if function in [SPATIAL_FN] and len(args) != 1:
            raise ValueError(
                f"in `{node.text}`, `{function}` requires one argument, got "
                f"this instead: {args}"
            )
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


def process_batch(
    batch_features, active_collection, batch_size, op_type, local_scale
):
    def reduce_image(image):
        reduced_points = image.reduceRegions(
            collection=batch_features,
            reducer=POINT_AGGREGATION_FUNCTIONS[op_type],
            scale=local_scale,
        )
        time_start_millis = image.get("system:time_start")
        return reduced_points.map(
            lambda feature: feature.set("system:time_start", time_start_millis)
        )

    batch_results = active_collection.map(reduce_image).flatten()
    return batch_results


def _scrub_key(key):
    key = key.replace(" ", "")
    key = re.sub(r"[,.());\\/]", "_", key)
    key = re.sub(r"_+", "_", key)
    key = key.replace("_nan", "")
    return key


# Define a function to expand year ranges into individual years
def expand_year_range(year_field):
    def _expand_year_range(row):
        if "-" in str(row[year_field]):
            start_year, end_year = map(int, row[year_field].split("-"))
            return [
                {**row, year_field: str(year)}
                for year in range(start_year, end_year + 1)
            ]

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
            closest_num = min(
                sorted_list[index - 1],
                sorted_list[index],
                key=lambda x: abs(x - num),
            )
        else:
            closest_num = sorted_list[index - 1]
        return closest_num


def chunk_points(point_list, chunk_size):
    """Yield successive chunks from point_list."""
    for i in range(0, len(point_list), chunk_size):
        yield point_list[i : i + chunk_size]


def initialize_gee():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(credentials_path)
    credentials = None
    if credentials_path:
        credentials = ee.ServiceAccountCredentials(None, credentials_path)
    ee.Initialize(
        credentials, opt_url="https://earthengine-highvolume.googleapis.com"
    )
    initalize_global_stat_functions()


def process_data_table(dataset_table_path, n_rows=None):
    dataset_table = pandas.read_csv(dataset_table_path, nrows=n_rows)

    dataset_table[SP_TM_AGG_OP] = None
    dataset_table[PIXEL_FN_OP] = None
    dataset_table[START_DATE] = None
    dataset_table[END_DATE] = None
    dataset_table[DATASET_TYPE] = None
    dataset_table[COLLECTION_TEMPORAL_RESOLUTION] = None
    dataset_table[NOMINAL_SCALE] = None

    missing_columns = set(EXPECTED_DATATABLE_COLUMNS).difference(
        set(dataset_table.columns)
    )
    if missing_columns:
        raise ValueError(
            "expected the following columns in the data table that were "
            + "missing:\n"
            + "\n\t".join(missing_columns)
            + "\nexisting columns:"
            + "\n\t".join(dataset_table.columns)
        )

    for row_index, dataset_row in dataset_table.iterrows():
        dataset_info = parse_gee_dataset_info(dataset_row[DATASET_ID])
        # Update DataFrame row with data parsed out of the URL/dataset id
        for key, value in dataset_info.items():
            dataset_table.at[row_index, key] = value
        if (
            isinstance(dataset_row[TRANSFORM_FUNC], str)
            and dataset_row[TRANSFORM_FUNC].lower() not in IGNORE_FLAGS
        ):
            transform_func_re = re.search(
                r"([^\[]*)(\[.*\])", dataset_row[TRANSFORM_FUNC]
            )
            if transform_func_re is None:
                raise ValueError(
                    f'"{dataset_row[TRANSFORM_FUNC]}" doesn\'t match any '
                    "Pixel Value Transform function"
                )
            pixel_fn, pixel_args = re.search(
                r"([^\[]*)(\[.*\])", dataset_row[TRANSFORM_FUNC]
            ).groups()
            if pixel_fn not in PIXEL_TRANSFORM_ALLOWED_FUNCTIONS:
                raise NotImplementedError(
                    f"{dataset_row[TRANSFORM_FUNC]} is not a valid pixel "
                    "transform, choose one of these "
                    "{PIXEL_TRANSFORM_ALLOWED_FUNCTIONS}"
                )
            dataset_table.at[row_index, PIXEL_FN_OP] = [
                pixel_fn,
                eval(pixel_args),
            ]
        spatiotemporal_fn = dataset_row[SP_TM_AGG_FUNC]
        spatiotemporal_fn = spatiotemporal_fn.replace(" ", "")
        try:
            grammar_tree = SPATIOTEMPORAL_FN_GRAMMAR.parse(spatiotemporal_fn)
        except Exception:
            raise ValueError(
                f'the function "{spatiotemporal_fn}" could not be parsed, check '
                f"for syntax errors\n error on row {dataset_row}"
            )
        lexer = SpatioTemporalFunctionProcessor()
        output = lexer.visit(grammar_tree)
        dataset_table.at[row_index, SP_TM_AGG_OP] = output
    try:
        dataset_table.to_csv("test.csv")
    except PermissionError:
        LOGGER.warning("test.csv is open, cannot overwrite")
    LOGGER.info(f"loaded {len(dataset_table)} datasets")
    return dataset_table


def generate_templates():
    for template_path, columns in [
        (DATATABLE_TEMPLATE_PATH, EXPECTED_DATATABLE_COLUMNS),
        (POINTTABLE_TEMPLATE_PATH, EXPECTED_POINTTABLE_COLUMNS),
    ]:
        if os.path.exists(template_path):
            LOGGER.warning(f"{template_path} already exists, not overwriting")
            continue
        with open(template_path, "w") as datatable:
            datatable.write(",".join(columns))
        LOGGER.info(f"wrote template to {template_path}")


INFER_RESOLUTION_MEMO_CACHE = {}
INFER_RESOLUTION_MEMO_LOCK = Lock()


def get_spatial_resolution(dataset_id):
    with INFER_RESOLUTION_MEMO_LOCK:
        if dataset_id not in INFER_RESOLUTION_MEMO_CACHE:
            asset_info = ee.data.getAsset(dataset_id)
            asset_type = asset_info.get("type", "").upper()

            # If not a collection, just handle it as a single image dataset
            if asset_type == "IMAGE_COLLECTION":
                collection = ee.ImageCollection(dataset_id)
                scale = collection.first().projection().nominalScale().getInfo()
            else:
                image = ee.Image(dataset_id)
                scale = image.projection().nominalScale().getInfo()

            INFER_RESOLUTION_MEMO_CACHE[dataset_id] = scale
        return INFER_RESOLUTION_MEMO_CACHE[dataset_id]


def get_year_julian_range(current_year, spatiotemporal_commands):
    """Returns offset of year in [min, max+1) range."""
    # Initialize current_year as the first day of this year
    year_range = (current_year, current_year + 1)
    julian_range = (1, 365)
    for spatiotemp_flag, op_type, args in spatiotemporal_commands:
        if spatiotemp_flag == YEARS_FN:
            year_range = (year_range[0] + args[0], year_range[1] + args[1])
        if spatiotemp_flag == JULIAN_FN:
            if len(args) == 2:
                # start and end date
                julian_range = tuple(args)
            else:
                # startss with an 'n'
                julian_range = tuple(args[1:3])
    return year_range, julian_range


def filter_imagecollection_by_date_range(
    year_range, julian_range, image_collection, bounding_box
):
    # Initialize an empty ImageCollection
    filtered_collection = ee.ImageCollection([])

    for year in range(year_range[0], year_range[1]):
        if julian_range[0] < 0:
            # For -1 to mean December 31 of the previous year, add 1 less than
            # the Julian day because -1 should add 0 days to December 31
            start_date = date(year - 1, 12, 31) + timedelta(
                days=julian_range[0] + 1
            )
        else:
            # Julian day 1 should be January 1st, so subtract 1 from the Julian day
            start_date = date(year, 1, 1) + timedelta(days=julian_range[0] - 1)

        if julian_range[1] > 365:
            # Subtract 365 to find the overflow and subtract an additional 1 since
            # January 1st should represent the 1st Julian day of the next year
            # but enddate is exclusive so set it off by 1 day
            end_date = date(year + 1, 1, 1) + timedelta(
                days=julian_range[1] - 365
            )
        else:
            # If within the same year, just adjust from January 1st
            end_date = date(year, 1, 1) + timedelta(days=julian_range[1])

        # Convert dates to strings
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Filter the image collection for the given range and merge with the cumulative collection
        year_filtered_collection = image_collection.filterDate(
            start_date_str, end_date_str
        ).filterBounds(bounding_box)
        filtered_collection = filtered_collection.merge(
            year_filtered_collection
        )
        size = filtered_collection.size().getInfo()
    return size, filtered_collection


LOAD_DATASET_MEMO_CACHE = {}
LOAD_DATASET_MEMO_LOCK = Lock()


def load_collection_or_image(dataset_id):
    with LOAD_DATASET_MEMO_LOCK:
        if dataset_id not in LOAD_DATASET_MEMO_CACHE:
            asset_info = ee.data.getAsset(dataset_id)
            asset_type = asset_info.get("type", "").upper()

            if asset_type == "IMAGE_COLLECTION":
                dataset = ee.ImageCollection(dataset_id)
            elif asset_type == "IMAGE":
                dataset = ee.ImageCollection([ee.Image(dataset_id)])
            else:
                # Fallback if not IMAGE or IMAGE_COLLECTION
                dataset = ee.ImageCollection([ee.Image(dataset_id)])

            LOAD_DATASET_MEMO_CACHE[dataset_id] = dataset
        return LOAD_DATASET_MEMO_CACHE[dataset_id]


def process_custom_dataset(
    dataset_id, point_features_by_year, point_unique_id_per_year, sp_tm_agg_op
):
    if dataset_id == "*GOOGLE/DYNAMICWORLD/V1 landcover table":
        return process_dynamicworld_landcover_table(
            point_features_by_year, point_unique_id_per_year, sp_tm_agg_op
        )
    elif dataset_id == "*MODIS/061/MCD12Q1 landcover table":
        return process_MODIS_landcover_table(
            point_features_by_year, point_unique_id_per_year, sp_tm_agg_op
        )


def process_dynamicworld_landcover_table(
    point_features_by_year, point_unique_id_per_year, sp_tm_agg_op
):
    results_by_key_class = {}
    dataset_info = parse_gee_dataset_info("GOOGLE/DYNAMICWORLD/V1")
    for lulc_class in range(9):
        key = f"GOOGLE/DYNAMICWORLD/V1_lulc_prop_area_{lulc_class}"
        lulc_point_id_value_list = process_gee_dataset(
            "GOOGLE/DYNAMICWORLD/V1",
            "label",
            dataset_info[START_DATE],
            dataset_info[END_DATE],
            dataset_info[COLLECTION_TEMPORAL_RESOLUTION],
            dataset_info[NOMINAL_SCALE],
            point_features_by_year,
            point_unique_id_per_year,
            (MASK_FN, [lulc_class]),
            sp_tm_agg_op,
        )
        results_by_key_class[key] = lulc_point_id_value_list
    return results_by_key_class


def process_MODIS_landcover_table(
    point_features_by_year, point_unique_id_per_year, sp_tm_agg_op
):
    results_by_lulc_class = {}
    for lulc_class in range(16):
        key = f"MODIS/061/MCD12Q1_lulc_prop_area_{lulc_class}"
        dataset_info = parse_gee_dataset_info("MODIS/061/MCD12Q1")
        lulc_point_id_value_list = process_gee_dataset(
            "MODIS/061/MCD12Q1",
            "LC_Type2",
            dataset_info[START_DATE],
            dataset_info[END_DATE],
            dataset_info[COLLECTION_TEMPORAL_RESOLUTION],
            dataset_info[NOMINAL_SCALE],
            point_features_by_year,
            point_unique_id_per_year,
            (MASK_FN, [lulc_class]),
            sp_tm_agg_op,
        )
        results_by_lulc_class[key] = lulc_point_id_value_list
    return results_by_lulc_class


def apply_time_filter(image_collection, year_range, julian_range, bounding_box):
    """Filter the ImageCollection to a year and julian date range within a bounding box."""
    active_size, active_collection = filter_imagecollection_by_date_range(
        year_range, julian_range, image_collection, bounding_box
    )
    return active_size, active_collection


def apply_pixel_transform(active_collection, pixel_op_transform):
    """Apply a pixel-level transform (mask, multiply, add, etc.)."""

    def pixel_op(pixel_op_fn, args):
        return {
            MASK_FN: lambda: active_collection.map(
                lambda img: img.remap(
                    ee.List(args), ee.List([1] * len(args)), 0
                ).copyProperties(img, ["system:time_start"])
            ),
            MULT_FN: lambda: active_collection.map(
                lambda img: img.multiply(ee.Image(args)).copyProperties(
                    img, ["system:time_start"]
                )
            ),
            ADD_FN: lambda: active_collection.map(
                lambda img: img.add(ee.Image(args)).copyProperties(
                    img, ["system:time_start"]
                )
            ),
        }.get(pixel_op_fn, lambda: None)()

    if pixel_op_transform is not None:
        pixel_op_fn, pixel_op_args = pixel_op_transform
        transformed_collection = pixel_op(pixel_op_fn, pixel_op_args)
        if transformed_collection is None:
            raise ValueError(f"Invalid pixel_op: {pixel_op_fn}")
        return transformed_collection
    return active_collection


def create_mask_collection(
    mask_dataset_id,
    mask_band_name,
    mask_codes,
    year_range,
    julian_range,
    bounding_box,
):
    """
    Build a 0/1 mask collection from a separate dataset based on integer codes.
    All pixels matching the `mask_codes` become 1, others become 0.
    """
    mask_base = load_collection_or_image(mask_dataset_id).select(mask_band_name)
    _, mask_filtered = apply_time_filter(
        mask_base, year_range, julian_range, bounding_box
    )
    return mask_filtered.map(
        lambda img: img.remap(
            ee.List(mask_codes), ee.List([1] * len(mask_codes)), 0
        ).copyProperties(img, ["system:time_start"])
    )


def combine_collections(base_collection, mask_collection):
    """
    Multiply base_collection by mask_collection, so 0-pixels from the mask
    will zero out the base while 1-pixels preserve the original values.
    """
    return ee.ImageCollection(
        base_collection.map(
            lambda base_img: base_img.multiply(
                mask_collection.filterDate(
                    base_img.get("system:time_start"),
                    base_img.get("system:time_start"),
                ).first()
            ).copyProperties(base_img, ["system:time_start"])
        )
    )


def apply_spatiotemporal_ops(
    active_collection,
    spatiotemporal_commands,
    collection_temporal_resolution,
    year_range,
    current_year,
    point_fc,
    bounding_buffer,
    nominal_scale,
):
    """
    Apply the user-defined commands (yearly aggregation, julian, spatial reduce, etc.)
    to the active_collection. Returns either an ImageCollection or a FeatureCollection
    depending on the final spatial reduce step.
    """
    applied_functions = set()
    for flag, op_type, args in spatiotemporal_commands:
        if flag in applied_functions:
            raise ValueError(f"{flag} was already applied.")
        applied_functions.add(flag)

        if isinstance(active_collection, ee.ImageCollection):
            if flag == YEARS_FN:
                # One image representing an aggregated year
                t_start = ee.Date.fromYMD(current_year, 1, 1).millis()
                agg_img = IMG_COL_AGGREGATION_FUNCTIONS[op_type](
                    active_collection
                )
                active_collection = ee.ImageCollection(
                    agg_img.set("system:time_start", t_start)
                )

            elif flag == JULIAN_FN:
                # Summarize over day-of-year range
                if op_type in N_LIMIT_OPS:
                    raise RuntimeError(f"Not implemented for images: {flag}")
                start_day, end_day = args

                def _op_by_julian_range(_year):
                    start_date = ee.Date.fromYMD(_year, 1, 1).advance(
                        start_day - 1, "day"
                    )
                    end_date = ee.Date.fromYMD(_year, 1, 1).advance(
                        end_day, "day"
                    )
                    daily_collection = active_collection.filterDate(
                        start_date, end_date
                    )
                    t_start = ee.Date.fromYMD(_year, 1, 1).millis()
                    return IMG_COL_AGGREGATION_FUNCTIONS[op_type](
                        daily_collection
                    ).set("system:time_start", t_start)

                active_collection = ee.ImageCollection.fromImages(
                    ee.List(list(range(year_range[0], year_range[1]))).map(
                        _op_by_julian_range
                    )
                )

            elif flag == SPATIAL_FN:
                if bounding_buffer > 0:
                    point_fc = point_fc.map(lambda f: f.buffer(bounding_buffer))
                local_scale = (
                    nominal_scale
                    if nominal_scale < bounding_buffer and bounding_buffer > 0
                    else DEFAULT_SCALE
                )

                def reduce_image(img):
                    reduced = img.reduceRegions(
                        collection=point_fc,
                        reducer=POINT_AGGREGATION_FUNCTIONS[op_type],
                        scale=local_scale,
                    )
                    return reduced.map(
                        lambda ft: ft.set(
                            "system:time_start", img.get("system:time_start")
                        )
                    )

                active_collection = active_collection.map(
                    reduce_image
                ).flatten()

        elif isinstance(active_collection, ee.FeatureCollection):
            # Handle FeatureCollection logic for YEARS_FN/JULIAN_FN
            pass

    return active_collection


def chunk_and_accumulate(fc, chunk_size, total_size):
    """Chunk a FeatureCollection and accumulate results into a list."""
    start = 0
    results = []
    while start < total_size:
        chunk = ee.FeatureCollection(fc.toList(chunk_size, start))
        partial_list = (
            chunk.map(
                lambda f: ee.Feature(
                    None,
                    {
                        UNIQUE_ID: f.get(UNIQUE_ID),
                        OUTPUT_TAG: f.get(OUTPUT_TAG),
                    },
                )
            )
            .reduceColumns(ee.Reducer.toList(2), [UNIQUE_ID, OUTPUT_TAG])
            .get("list")
        )
        results.extend(ee.List(partial_list).getInfo())
        start += chunk_size
    return results


def process_gee_dataset(
    dataset_id,
    band_name,
    dataset_start_date,
    dataset_end_date,
    collection_temporal_resolution,
    nominal_scale,
    point_list_by_year,
    point_unique_id_per_year,
    pixel_op_transform,
    spatiotemporal_commands,
    mask_dataset_id=None,
    mask_band_name=None,
    mask_codes=None,
):
    # Ensure SPATIAL_FN is present
    if SPATIAL_FN not in [x[0] for x in spatiotemporal_commands]:
        spatiotemporal_commands += [(SPATIAL_FN, MEAN_STAT, [0])]
        bounding_buffer = 0
    else:
        bounding_buffer = next(
            x[2][0] for x in spatiotemporal_commands if x[0] == SPATIAL_FN
        )

    # Parse dataset date range
    try:
        dataset_start_year = int(dataset_start_date[:4])
        dataset_end_year = int(dataset_end_date[:4])
    except ValueError:
        dataset_start_year, dataset_end_year = 0, 9999

    # Load main dataset
    base_collection = load_collection_or_image(dataset_id).select(band_name)

    results_by_year = collections.defaultdict(list)

    for current_year_batch_id in point_list_by_year.keys():
        if "_" in current_year_batch_id:
            current_year = int(current_year_batch_id.split("_")[0])
        else:
            current_year = int(current_year_batch_id)

        year_range, julian_range = get_year_julian_range(
            current_year, spatiotemporal_commands
        )
        if not (
            dataset_start_year <= year_range[0] <= dataset_end_year
        ) or not (dataset_start_year <= year_range[1] <= dataset_end_year):
            results_by_year[current_year_batch_id] = [
                (
                    uid,
                    f"Outside {dataset_start_year}-{dataset_end_year}: {year_range}",
                )
                for uid in point_unique_id_per_year[current_year_batch_id]
            ]
            continue

        point_fc = ee.FeatureCollection(
            point_list_by_year[current_year_batch_id]
        )
        bounding_box = point_fc.geometry().bounds()
        if bounding_buffer > 0:
            bounding_box = bounding_box.buffer(bounding_buffer)

        active_size, active_collection = apply_time_filter(
            base_collection, year_range, julian_range, bounding_box
        )
        active_collection = apply_pixel_transform(
            active_collection, pixel_op_transform
        )

        if mask_dataset_id and mask_band_name and mask_codes:
            mask_full = load_collection_or_image(mask_dataset_id).select(
                mask_band_name
            )
            _, mask_collection = filter_imagecollection_by_date_range(
                year_range, [1, 365], mask_full, bounding_box
            )

            mask_collection = mask_collection.map(
                lambda m: m.remap(
                    ee.List(mask_codes), ee.List([1] * len(mask_codes)), 0
                )
            )
            yearly_mask_image = mask_collection.reduce(ee.Reducer.mode())

            active_collection = active_collection.map(
                lambda img: img.multiply(yearly_mask_image).copyProperties(
                    img, ["system:time_start"]
                )
            )

        # Apply spatiotemporal commands (yearly/julian/spatial reduce)
        active_collection = apply_spatiotemporal_ops(
            active_collection,
            spatiotemporal_commands,
            collection_temporal_resolution,
            year_range,
            current_year,
            point_fc,
            bounding_buffer,
            nominal_scale,
        )

        # Now batch and accumulate if we have a FeatureCollection
        if isinstance(active_collection, ee.FeatureCollection):
            active_collection_size = active_size
            results = chunk_and_accumulate(
                active_collection, BATCH_SIZE, active_collection_size
            )
            results_by_year[current_year_batch_id].extend(results)
        else:
            results_by_year[current_year_batch_id].extend([])

    # Flatten results
    final_list = []
    for year_batch, data in results_by_year.items():
        final_list.extend([(item[0], item[1]) for item in data])

    return final_list


def _debug_save_image(active_collection, desc):
    half_side_length = 0.1  # Change this to half of your desired side length
    center_lng = -72.5688086775836
    center_lat = 42.43584566068613
    coordinates = [
        [
            center_lng - half_side_length,
            center_lat - half_side_length,
        ],  # Lower-left corner
        [
            center_lng + half_side_length,
            center_lat - half_side_length,
        ],  # Lower-right corner
        [
            center_lng + half_side_length,
            center_lat + half_side_length,
        ],  # Upper-right corner
        [
            center_lng - half_side_length,
            center_lat + half_side_length,
        ],  # Upper-left corner
        [
            center_lng - half_side_length,
            center_lat - half_side_length,
        ],  # Closing the loop
    ]

    LOGGER.debug(active_collection.getInfo())
    image = ee.Image(active_collection.first())
    LOGGER.debug(f"scale: {image.projection().nominalScale().getInfo()}")
    LOGGER.debug(image.getInfo())
    square = ee.Geometry.Polygon([coordinates])
    export = ee.batch.Export.image.toDrive(
        image=image, description=desc, scale=30, region=square
    )  # Define the region
    export.start()
    LOGGER.debug(export.status())


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="sample points on GEE data")
    parser.add_argument(
        "--point_table_paths",
        required=True,
        nargs="+",
        help=f"paths to tables with {LAT_FIELD}, {LNG_FIELD}, and {YEAR_FIELD} columns (space-separated)",
    )
    parser.add_argument(
        "--dataset_table_path", required=True, help="path to data table"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"how many points to process per batch, defaults to {BATCH_SIZE}",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help=f"how many datasets to process in parallel, defaults to {MAX_WORKERS}",
    )
    parser.add_argument(
        "--n_point_table_rows",
        type=int,
        help="Limit the number of points to this many rows for testing.",
    )
    parser.add_argument(
        "--n_dataset_rows",
        type=int,
        help="Limit the dataset table to read this many rows",
    )

    args = parser.parse_args()
    initialize_gee()

    # point_features_by_year, point_unique_id_per_year, point_table = (
    #     point_table_to_point_batch(
    #         args.point_table_path,
    #         n_rows=args.n_point_table_rows))

    dataset_table = process_data_table(
        args.dataset_table_path, n_rows=args.n_dataset_rows
    )

    # Process each point table path
    for point_table_path in args.point_table_paths:
        # Load point table and prepare for processing
        point_features_by_year, point_unique_id_per_year, point_table = (
            point_table_to_point_batch(
                point_table_path, n_rows=args.n_point_table_rows
            )
        )

        for row_index, dataset_row in dataset_table.iterrows():
            key = (
                f"{dataset_row[DATASET_ID]}_"
                f"{dataset_row[BAND_NAME]}_"
                f"{dataset_row[SP_TM_AGG_FUNC]}_"
                f"{dataset_row[TRANSFORM_FUNC]}"
            )
            LOGGER.debug(f"************ this is the key: {key}")
            if key not in point_table.columns:
                point_table[key] = None  # Ensure column exists
            point_id_value_list = process_gee_dataset(
                dataset_row[DATASET_ID],
                dataset_row[BAND_NAME],
                dataset_row[START_DATE],
                dataset_row[END_DATE],
                dataset_row[COLLECTION_TEMPORAL_RESOLUTION],
                dataset_row[NOMINAL_SCALE],
                point_features_by_year,
                point_unique_id_per_year,
                dataset_row[PIXEL_FN_OP],
                dataset_row[SP_TM_AGG_OP],
            )

            for row_id, value in point_id_value_list:
                if row_id in point_table.index:
                    point_table.at[row_id, key] = value

        # Save the updated point table with a timestamped filename
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f'{os.path.basename(point_table_path).rsplit(".", 1)[0]}_{timestamp}.csv'
        point_table.to_csv(output_filename, index=False)
        LOGGER.info(f"Saved updated point table to {output_filename}")


if __name__ == "__main__":
    main()
