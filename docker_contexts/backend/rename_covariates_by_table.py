"""Used to rename elemenets in database given the value of the first covariate"""
import argparse
import csv

import pandas as pd
from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, update


def load_corrections_from_csv(file_path):
    df = pd.read_csv(file_path)
    corrections = list(zip(df['Misspelled'], df['Correct spelling']))
    single_word_corrections = [
        (x.split(' ')[0], y.split(' ')[0]) for x, y in corrections]
    return single_word_corrections


def update_response_variables(session):
    # Fetch covariate_defn_ids for 'Crop_latin_name' and 'Crop_common_name'
    rows = [
        {
            'Crop_latin_name': 'brassica oleracea, broccoli',
            'rename': 'brassica oleracea',
            'Crop_common_name': 'broccoli'
        },
        {
            'Crop_latin_name': 'brassica oleracea, broccoli, cauliflower, cabbage',
            'rename': 'brassica oleracea',
            'Crop_common_name': 'cole crops'
        },
        {
            'Crop_latin_name': 'grassland',
            'rename': None,
            'Crop_common_name': 'grassland'
        },
    ]
    defn_ids = dict(session.query(CovariateDefn.name, CovariateDefn.id_key).filter(
        CovariateDefn.name.in_(['Crop_latin_name', 'Crop_common_name'])
    ).all())
    crop_latin_name_defn_id = defn_ids['Crop_latin_name']
    crop_common_name_defn_id = defn_ids['Crop_common_name']

    for row in rows:
        crop_latin_name_value = row['Crop_latin_name']
        rename_value = row['rename']
        crop_common_name_value = row['Crop_common_name']

        # Find sample_ids where Crop_latin_name matches
        sample_ids = [
            s[0] for s in session.query(CovariateValue.sample_id).filter(
                CovariateValue.covariate_defn_id == crop_latin_name_defn_id,
                CovariateValue.value == crop_latin_name_value
            ).all()
        ]

        if not sample_ids:
            continue  # No matching samples, proceed to next row

        # Update or delete Crop_latin_name based on 'rename' value
        if rename_value:
            session.query(CovariateValue).filter(
                CovariateValue.covariate_defn_id == crop_latin_name_defn_id,
                CovariateValue.sample_id.in_(sample_ids)
            ).update(
                {CovariateValue.value: rename_value},
                synchronize_session=False
            )
        else:
            session.query(CovariateValue).filter(
                CovariateValue.covariate_defn_id == crop_latin_name_defn_id,
                CovariateValue.sample_id.in_(sample_ids)
            ).delete(synchronize_session=False)

        # Update existing Crop_common_name values
        session.query(CovariateValue).filter(
            CovariateValue.covariate_defn_id == crop_common_name_defn_id,
            CovariateValue.sample_id.in_(sample_ids)
        ).update(
            {CovariateValue.value: crop_common_name_value},
            synchronize_session=False
        )

        # Find sample_ids without Crop_common_name
        existing_common_name_sample_ids = {
            s[0] for s in session.query(CovariateValue.sample_id).filter(
                CovariateValue.covariate_defn_id == crop_common_name_defn_id,
                CovariateValue.sample_id.in_(sample_ids)
            ).all()
        }
        missing_sample_ids = set(sample_ids) - existing_common_name_sample_ids

        # Insert new Crop_common_name covariate values where missing
        new_covariates = [
            CovariateValue(
                sample_id=sample_id,
                covariate_defn_id=crop_common_name_defn_id,
                value=crop_common_name_value
            )
            for sample_id in missing_sample_ids
        ]
        session.bulk_save_objects(new_covariates)

    session.commit()


    # # Iterate through each row in the corrections table

    # for row in [
    #         {
    #             'Crop_latin_name': 'brassica oleracea, broccoli',
    #             'rename': 'brassica oleracea',
    #             'Crop_common_name': 'broccoli'
    #         },
    #         {
    #             'Crop_latin_name': 'brassica oleracea, broccoli, cauliflower, cabbage',
    #             'rename': 'brassica oleracea',
    #             'Crop_common_name': 'cole crops'
    #         },
    #         {
    #             'Crop_latin_name': 'grassland',
    #             'rename': None,
    #             'Crop_common_name': 'grassland'
    #         },]:
    #     # sampling_method = row['Sampling_method']
    #     # units = row['Units']
    #     # corrected_response_variable = row['CORRECTED: Response_variable']
    #     # Get sample IDs where Sampling_method and Units match
    #     crop_latin_name = row['Crop_latin_name']
    #     subquery = (
    #         session.query(CovariateValue.sample_id)
    #         .join(CovariateDefn)
    #         .filter(
    #             CovariateDefn.name == "Crop_latin_name",
    #             CovariateValue.value == crop_latin_name
    #         )
    #     ).subquery()

    #     values = session.execute(
    #         select(CovariateValue)
    #         .join(CovariateDefn)
    #         .filter(
    #             CovariateDefn.name == "Crop_common_name",
    #             CovariateValue.sample_id.in_(subquery)
    #         )
    #     )





    #     print([x[0].value for x in values.all()])
    #     continue

    #     # Update Response_variable values for matching sample IDs
    #     session.execute(
    #         update(CovariateValue)
    #         .where(
    #             CovariateValue.sample_id.in_(subquery),
    #             CovariateValue.covariate_defn_id == session.query(CovariateDefn.id_key)
    #             .filter(CovariateDefn.name == "Response_variable")
    #             .scalar_subquery()
    #         )
    #         .values(value=corrected_response_variable)
    #     )
    # session.commit()


# def update_covariate_values(session, corrections):
#     for misspelled, correct in corrections:
#         stmt = (
#             update(CovariateValue)
#             .where(
#                 CovariateValue.value == misspelled,
#                 CovariateValue.covariate_defn.has(CovariateDefn.name == 'Genus')
#             )
#             .values(value=correct)
#         )
#         result = session.execute(stmt)
#         if result.rowcount > 0:
#             print(f"Updated {result.rowcount} entries: '{misspelled}' -> '{correct}'")

#     # Commit the changes
#     session.commit()
#     print("Updates completed successfully.")


def main():
    init_db()
    parser = argparse.ArgumentParser(description='rename or add covariates by table defn')
    #parser.add_argument('table_path')
    args = parser.parse_args()

    #corrections_list = load_corrections_from_csv(args.table_path)
    session = SessionLocal()
    update_response_variables(session)
    #update_covariate_values(session, corrections_list)
    session.close()


if __name__ == '__main__':
    main()
