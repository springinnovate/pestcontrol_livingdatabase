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


def update_response_variables(session, corrections):
    """
    Update Response_variable covariate values based on Sampling_method and Units.

    Args:
        session (Session): SQLAlchemy session.
        corrections (list): List of corrections with each row containing:
                            - Sampling_method
                            - Units
                            - Corrected Response_variable
    """
    # Iterate through each row in the corrections table
    for index, row in corrections.iterrows():
        sampling_method = row['Sampling_method']
        units = row['Units']
        corrected_response_variable = row['CORRECTED: Response_variable']

        print(f'{index}: {sampling_method}, {units}, {corrected_response_variable}')
        # Get sample IDs where Sampling_method and Units match
        subquery = (
            session.query(CovariateValue.sample_id)
            .join(CovariateDefn)
            .filter(
                CovariateDefn.name == "Sampling_method",
                CovariateValue.value == sampling_method
            )
            .intersect(
                session.query(CovariateValue.sample_id)
                .join(CovariateDefn)
                .filter(
                    CovariateDefn.name == "Units",
                    CovariateValue.value == units
                )
            )
        ).subquery()

        values = session.execute(
            select(CovariateValue)
            .join(CovariateDefn)
            .filter(
                CovariateDefn.name == "Response_variable",
                CovariateValue.sample_id.in_(subquery)
            )
        )
        print([x[0].value for x in values.all()])
        continue

        # Update Response_variable values for matching sample IDs
        session.execute(
            update(CovariateValue)
            .where(
                CovariateValue.sample_id.in_(subquery),
                CovariateValue.covariate_defn_id == session.query(CovariateDefn.id_key)
                .filter(CovariateDefn.name == "Response_variable")
                .scalar_subquery()
            )
            .values(value=corrected_response_variable)
        )
    session.commit()


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
    corrections_table = pd.read_csv(r"C:\Users\richp\Downloads\activity_corrections.csv")
    session = SessionLocal()
    update_response_variables(session, corrections_table)
    #update_covariate_values(session, corrections_list)
    session.close()


if __name__ == '__main__':
    main()
