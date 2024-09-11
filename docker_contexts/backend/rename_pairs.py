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
from database import SessionLocal, init_db, backup_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn, CovariateAssociation
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import aliased
from sqlalchemy import update
from sqlalchemy.orm import selectinload

BLANK = '*BLANK*'


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
        if not cov_a_value:
            continue
        if cov_b_value == BLANK:
            continue
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
    df = pd.read_csv(table_path, usecols=[0, 1], skiprows=1, header=None, on_bad_lines='skip')
    first_two_columns = df.iloc[:, :2].dropna().values.tolist()
    print(first_two_columns)
    return first_two_columns


def main():
    init_db()
    parser = argparse.ArgumentParser(description='covariate pairs')
    parser.add_argument('cov_name_a')
    parser.add_argument('cov_name_b')
    parser.add_argument('--rename_table_path')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    table_path = f'rename_table_{args.cov_name_a}_xxx_{args.cov_name_b}.csv'
    if os.path.exists(table_path) and not (args.force or args.rename_table_path):
        print(f'WARNING: "{table_path}" exists, exiting now. Re-run with the `--force` flag to overwrite')
        sys.exit(-1)

    if args.rename_table_path:
        backup_db()
        covariate_pairs = load_covariate_pairs(args.rename_table_path)
        update_cov_values(args.cov_name_a, args.cov_name_b, covariate_pairs)
        print('all updated')
        return

    # Create aliases for the two covariates
    cov_a = aliased(CovariateValue)
    cov_b = aliased(CovariateValue)

    session = SessionLocal()
    cov_a_defn = session.execute(
        select(CovariateDefn)
        .filter(CovariateDefn.name == args.cov_name_a)).scalar_one_or_none()
    cov_b_defn = session.execute(
        select(CovariateDefn)
        .filter(CovariateDefn.name == args.cov_name_b)).scalar_one_or_none()

    print(f'{args.cov_name_a}: {cov_a_defn}; {args.cov_name_b}: {cov_b_defn}')

    if not cov_a_defn.covariate_association == cov_b_defn.covariate_association:
        raise RuntimeError(f'"{args.cov_name_a}" and "{args.cov_name_b}"" are different covariate types (one is Study other is Sample)')

    missing = False
    for name, cov in [(args.cov_name_a, cov_a_defn), (args.cov_name_b, cov_b_defn)]:
        if cov is None:
            print(f'error, "{name}" is not a covariate in the database')
            missing = True
    if missing:
        print('Exiting, fix covariate name issue.')
        sys.exit(-1)

    if cov_a_defn.covariate_association == CovariateAssociation.STUDY.value:
        print('these are STUDY level covariate')
        same_type_filter = cov_a.study_id == cov_b.study_id
    elif cov_a_defn.covariate_association == CovariateAssociation.SAMPLE.value:
        print('these are SAMPLE level covariates')
        same_type_filter = cov_a.study_id == cov_b.study_id

    query = (
        select(cov_a.value, cov_b.value)  # Select cov_a and cov_b values
        .join(CovariateDefn, CovariateDefn.id_key == cov_a.covariate_defn_id)  # Join CovariateDefn for cov_a
        .outerjoin(cov_b, and_(
            same_type_filter,
            cov_b.covariate_defn_id == cov_b_defn.id_key
        ))
        .filter(cov_a.covariate_defn_id == cov_a_defn.id_key)  # Ensure cov_a matches the correct definition
    ).distinct()
    results = session.execute(query)

    unique_covariate_pairs = collections.defaultdict(list)
    for col_a, col_b in results:
        unique_covariate_pairs[col_a].append(col_b if col_b is not None else BLANK)

    # Prepare the data for DataFrame
    data = []
    max_len = 0

    for col_a, col_b_list in sorted(unique_covariate_pairs.items()):
        max_len = max(max_len, len(col_b_list))  # Find the longest col_b_list
        data.append([col_a] + col_b_list)

    # Create a list of column names: first two are defined, rest are generic ('col_n')
    columns = [args.cov_name_a, args.cov_name_b] + [f'col_{i}' for i in range(2, max_len+1)]

    # Create the DataFrame, filling missing values with NaN where col_b_list is shorter
    df = pd.DataFrame(data, columns=columns)
    df.fillna('', inplace=True)
    df.to_csv(table_path, index=False)

    print(f'written to {table_path}')


if __name__ == '__main__':
    main()
