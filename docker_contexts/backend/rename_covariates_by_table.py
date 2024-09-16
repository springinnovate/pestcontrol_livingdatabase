import collections
import argparse
import sys, os


import pandas as pd
from database import SessionLocal, init_db, backup_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn, CovariateAssociation
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import aliased
from sqlalchemy import update, text
from sqlalchemy.orm import selectinload


def main():
    init_db()
    parser = argparse.ArgumentParser(description='rename or add covariates by table defn')
    parser.add_argument('table_path')
    args = parser.parse_args()

    df = pd.read_csv(args.table_path)
    session = SessionLocal()

    covariate_defn_list = session.execute(
        select(CovariateDefn)).all()
    covariate_defn_lookup = {
        covariate_defn[0].name: covariate_defn[0]
        for covariate_defn in covariate_defn_list
    }
    base_name = df.columns[0]
    base_name_defn = covariate_defn_lookup.get(base_name)

    unique_base_values = df[base_name].unique()
    matching_sample_ids_subquery = (
        select(CovariateValue.sample_id)
        .where(
            CovariateValue.covariate_defn_id == base_name_defn.id_key,
            CovariateValue.value.in_(unique_base_values),
            CovariateValue.sample_id.isnot(None)
        )
        .subquery()
    )

    covariate_values_query = (
        select(CovariateValue)
        .where(
            CovariateValue.sample_id.in_(matching_sample_ids_subquery.select()),
            CovariateValue.covariate_defn_id.in_([defn.id_key for defn in covariate_defn_lookup.values()])
        )
    )

    print('calculating covariate value list')
    covariate_values_list = session.execute(covariate_values_query).scalars().all()

    covariate_value_lookup = {}
    for cov_value in covariate_values_list:
        covariate_value_lookup.setdefault(cov_value.sample_id, {})[cov_value.covariate_defn_id] = cov_value

    print('calculating samples to process')
    samples_to_process = (
        select(Sample.id_key, CovariateValue.value)
        .join(CovariateValue)
        .where(
            CovariateValue.covariate_defn_id == base_name_defn.id_key,
            CovariateValue.value.in_(unique_base_values)
        )
    )
    samples_to_process_list = session.execute(samples_to_process).all()

    base_value_to_sample_ids = {}
    for sample_id, value in samples_to_process_list:
        base_value_to_sample_ids.setdefault(value, []).append(sample_id)

    new_covariate_values = []
    print(len(samples_to_process_list))
    return
    # Iterate over rows in the dataframe
    for row_id, row in df.iterrows():
        base_value = row[base_name]
        covariate_mappings = row.drop(labels=base_name).to_dict()

        sample_ids = base_value_to_sample_ids.get(base_value, [])
        if not sample_ids:
            print(f"No samples found for base_value '{base_value}'")
            continue

        for sample_id in sample_ids:
            print(f'{base_name}: {base_value}')
            for covariate_name, new_value in covariate_mappings.items():
                covariate_defn = covariate_defn_lookup[covariate_name]

                sample_covariates = covariate_value_lookup.get(sample_id, {})
                covariate_value = sample_covariates.get(covariate_defn.id_key)

                if covariate_value:
                    if covariate_value.value != new_value:
                        print(
                            f"Updating sample_id {sample_id}: "
                            f"'{covariate_name}' from "
                            f"'{covariate_value.value}' to '{new_value}'")
                        covariate_value.value = new_value
                else:
                    print(
                        f"Adding new CovariateValue for sample_id "
                        f"{sample_id}: '{covariate_name}' = '{new_value}'")
                    new_covariate_value = CovariateValue(
                        sample_id=sample_id,
                        covariate_defn_id=covariate_defn.id_key,
                        value=new_value
                    )
                    new_covariate_values.append(new_covariate_value)
                    # Update the lookup to include this new covariate_value
                    covariate_value_lookup.setdefault(sample_id, {})[
                        covariate_defn.id_key] = new_covariate_value
            break
        break
    # if new_covariate_values:
    #     session.bulk_save_objects(new_covariate_values)

    # # Commit all changes
    # session.commit()
    # session.close()


if __name__ == '__main__':
    main()
