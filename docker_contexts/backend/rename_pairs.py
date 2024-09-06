"""This is used in cases where we have
column_a - column_b

and there are multiple pairs of covariate a that
match covariate b in a sample

i.e. 'Crop_latin_name' -- 'Crop_common_name'

if there's samples where there is a covariate a but not a b
"""
import collections
import argparse
import sys, os


import pandas as pd
from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, func
from sqlalchemy.orm import aliased
from sqlalchemy import update


def update_cov_values(cov_a_name, cov_b_name, covariate_pairs):
    session = SessionLocal()

    cov_b_defn = session.query(CovariateDefn).filter(CovariateDefn.name == cov_b_name).first()

    cov_a_values = [pair[0] for pair in covariate_pairs]
    samples_with_cov_a = (
        session.query(Sample, CovariateValue.value)
        .join(CovariateDefn, CovariateValue.covariate_defn_id == CovariateDefn.id_key)
        .join(Sample, CovariateValue.sample_id == Sample.id_key)
        .filter(CovariateDefn.name == cov_a_name)
        .filter(CovariateValue.value.in_(cov_a_values))  # Match any cov_a_value in the pairs
        .all()
    )

    samples_by_cov_a_value = collections.defaultdict(list)
    for sample, cov_a_value in samples_with_cov_a:
        samples_by_cov_a_value[cov_a_value].append(sample)

    for cov_a_value, cov_b_value in covariate_pairs:
        print(f'Processing {cov_a_value} -> {cov_b_value}')

        # Get all samples for this cov_a_value
        samples = samples_by_cov_a_value[cov_a_value]

        # Update or insert cov_b_value for each sample
        existing_cov_b_values = (
            session.query(CovariateValue)
            .filter(CovariateValue.sample_id.in_([sample.id_key for sample in samples]))
            .filter(CovariateValue.covariate_defn_id == cov_b_defn.id_key)
            .all()
        )
        existing_cov_b_map = {cov_b.sample_id: cov_b for cov_b in existing_cov_b_values}

        # Step 3: Prepare lists for bulk update and insert
        update_mappings = []
        insert_mappings = []

        for sample in samples:
            if sample.id_key in existing_cov_b_map:
                # Prepare for update
                update_mappings.append({
                    'id_key': existing_cov_b_map[sample.id_key].id_key,  # Primary key to update
                    'value': cov_b_value
                })
            else:
                # Prepare for insert
                insert_mappings.append({
                    'value': cov_b_value,
                    'covariate_defn_id': cov_b_defn.id_key,
                    'sample_id': sample.id_key
                })

        # Step 4: Perform bulk update
        if update_mappings:
            session.bulk_update_mappings(CovariateValue, update_mappings)

        # Step 5: Perform bulk insert
        if insert_mappings:
            session.bulk_insert_mappings(CovariateValue, insert_mappings)

    session.commit()


def load_covariate_pairs(table_path):
    df = pd.read_csv(table_path, usecols=[0, 1], header=None, on_bad_lines='skip')
    first_two_columns = df.iloc[:, :2].dropna().values.tolist()
    print(first_two_columns)
    return first_two_columns

def main():
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='covariate pairs')
    parser.add_argument('cov_name_a')
    parser.add_argument('cov_name_b')
    parser.add_argument('--rename_table_path')
    args = parser.parse_args()

    if args.rename_table_path:
        covariate_pairs = load_covariate_pairs(args.rename_table_path)
        update_cov_values(args.cov_name_a, args.cov_name_b, covariate_pairs)
        print('all updated')
        return

    # Create aliases for the two covariates
    cov_a = aliased(CovariateValue)
    cov_b = aliased(CovariateValue)

    subquery = (
        select(cov_a.sample_id)
        .join(CovariateDefn, CovariateDefn.id_key == cov_a.covariate_defn_id)
        .filter(CovariateDefn.name == args.cov_name_a)
    )

    # Main query to get the values for cov_name_a and cov_name_b together
    query = (
        select(cov_a.value, cov_b.value)
        .join(CovariateDefn, CovariateDefn.id_key == cov_a.covariate_defn_id)
        .outerjoin(cov_b, cov_b.sample_id == cov_a.sample_id)  # Left outer join to include cases where cov_b may not exist
        .filter(CovariateDefn.name == args.cov_name_a)
        .filter(cov_b.covariate_defn.has(CovariateDefn.name == args.cov_name_b))
        .filter(cov_a.sample_id.in_(subquery))
    ).distinct()

    results = session.execute(query).all()
    unique_covariate_pairs = collections.defaultdict(list)
    for col_a, col_b in sorted(results):
        unique_covariate_pairs[col_a].append(col_b)
    table_path = 'rename_table.csv'
    with open(table_path, 'w') as table:
        for col_a, col_b_list in sorted(unique_covariate_pairs.items()):
            table.write(f'{col_a},' + ','.join(col_b_list) + '\n')
    print(f'written to {table_path}')
if __name__ == '__main__':
    main()
