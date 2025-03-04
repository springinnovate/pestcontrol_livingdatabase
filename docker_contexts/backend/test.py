import ee
from gee_database_point_sampler import process_gee_dataset
from gee_database_point_sampler import parse_spatiotemporal_fn
from gee_database_point_sampler import initialize_gee
from gee_database_point_sampler import UNIQUE_ID

def main():
    initialize_gee()
    dataset_id = 'ECMWF/ERA5/MONTHLY'
    band_name = 'dewpoint_2m_temperature'
    dataset_start_date = '0'
    dataset_end_date = '9999'
    collection_temporal_resolution = 'years'
    nominal_scale = 1000

    # point_features_by_year[batch_key].append(
    #         ee.Feature(ee.Geometry.Point(
    #             [row[LNG_FIELD], row[LAT_FIELD]], 'EPSG:4326'),
    #             {UNIQUE_ID: index}))
    point_list_by_year = {
        '2006': [ee.Feature(ee.Geometry.Point([x[1], x[0]], 'EPSG:4326'), {UNIQUE_ID: index}) for index, x in enumerate([(36.12661, -119.936328), (36.214052, -119.927745), (36.126714, -119.945297)])]
    }
    point_unique_id_per_year = {
        '2006': [0, 1, 2]
    }
    pixel_op_transform = None
    spatiotemporal_commands = (
        'spatial_min(30;years_mean(-1,0;julian_min(90, 180)))'
    )
    mask_dataset_id = None
    mask_band_name = None
    mask_codes = None
    result = process_gee_dataset(
        dataset_id,
        band_name,
        dataset_start_date,
        dataset_end_date,
        collection_temporal_resolution,
        nominal_scale,
        point_list_by_year,
        point_unique_id_per_year,
        pixel_op_transform,
        parse_spatiotemporal_fn(spatiotemporal_commands),
        mask_dataset_id=None,
        mask_band_name=None,
        mask_codes=None)

    print(result)

if __name__ == '__main__':
    main()
