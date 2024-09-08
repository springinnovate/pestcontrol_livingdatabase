"""This is used to query a column and try to correlated similarly spelled values."""
import os
import collections
import argparse

import pandas as pd
from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, delete

DELETE = 'DELETE'


def load_covariate_pairs(table_path):
    df = pd.read_csv(table_path, usecols=[0, 1], skiprows=1, header=None, on_bad_lines='skip')
    first_two_columns = df.iloc[:, :2].dropna().values.tolist()
    print(first_two_columns)
    return first_two_columns


def main():
    init_db()
    parser = argparse.ArgumentParser(description='Drop covariate by name')
    parser.add_argument('covariate_name')
    parser.add_argument('--verify')
    args = parser.parse_args()

    session = SessionLocal()
    covariate_defn = session.execute(
        select(CovariateDefn)
        .filter(CovariateDefn.name == args.covariate_name)).scalar_one_or_none()
    if covariate_defn is None:
        print(f'could not find a covariate named {args.covariate_name}')
        return

    table_path = f'drop_covariates_{args.covariate_name}.csv'

    if os.path.exists(table_path):
        df = pd.read_csv(table_path)
        column_name = df.columns[0]
        if column_name == DELETE:
            print(f'deleting {args.covariate_name}')
            delete_query = delete(CovariateValue).where(CovariateValue.covariate_defn_id == covariate_defn.id_key)
            session.execute(delete_query)
            session.delete(covariate_defn)
            session.commit()
            print(f'deleted all {args.covariate_name} covariates and the definition')
            return
        else:
            print(
                f'ERROR was expecting if you wanted to delete {args.covariate_name} you '
                f'would have replaced the columname at {table_path} with {DELETE}, instead '
                f'it is "{column_name}"')
        return


    print(covariate_defn)
    covariate_values = session.execute(
        select(CovariateValue.value)
        .filter(CovariateValue.covariate_defn_id == covariate_defn.id_key).distinct())
    covariate_name_list = [x[0] for x in covariate_values.all()]

    df = pd.DataFrame()
    df[f'{args.covariate_name} (replace this with {DELETE} if you want to delete it)'] = covariate_name_list
    df.to_csv(table_path, index=False)
    print(f'okay, check out {table_path} and make sure it is what you want to delete and follow the directions there and run this script again')
    return

    query = (
        select(CovariateValue.value)
        .join(CovariateDefn)
        .filter(CovariateDefn.name == args.covariate_name)
    ).distinct()
    results = session.execute(query).all()
    for result in results:
        print(result)
    return

    unique_covariate_pairs = collections.defaultdict(list)
    for col_a, col_b in results:
        unique_covariate_pairs[col_a].append(col_b if col_b is not None else BLANK)
    table_path = f'rename_table_{args.cov_name_a}_xxx_{args.cov_name_b}.csv'

    # Prepare the data for DataFrame
    data = []
    max_len = 0  # To track the maximum number of elements in col_b_list

    for col_a, col_b_list in sorted(unique_covariate_pairs.items()):
        max_len = max(max_len, len(col_b_list))  # Find the longest col_b_list
        data.append([col_a] + col_b_list)

    # Create a list of column names: first two are defined, rest are generic ('col_n')
    columns = [args.cov_name_a, args.cov_name_b] + [f'col_{i}' for i in range(2, max_len)]

    # Create the DataFrame, filling missing values with NaN where col_b_list is shorter
    df = pd.DataFrame(data, columns=columns)
    df.fillna('', inplace=True)
    df.to_csv(table_path, index=False)

    print(f'written to {table_path}')
if __name__ == '__main__':
    main()
