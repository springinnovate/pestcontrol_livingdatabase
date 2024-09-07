"""This is used to query a column and try to correlated similarly spelled values."""
import collections
import argparse

import pandas as pd
from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import aliased
from sqlalchemy import update
from sqlalchemy.orm import selectinload

BLANK = '*BLANK*'


def load_covariate_pairs(table_path):
    df = pd.read_csv(table_path, usecols=[0, 1], skiprows=1, header=None, on_bad_lines='skip')
    first_two_columns = df.iloc[:, :2].dropna().values.tolist()
    print(first_two_columns)
    return first_two_columns


def main():
    init_db()
    parser = argparse.ArgumentParser(description='Similar covariate pairs')
    parser.add_argument('covariate_name')
    parser.add_argument('--rename_table_path')
    args = parser.parse_args()

    if args.rename_table_path:
        print('not implemented')
        return

    session = SessionLocal()
    # covariate_defn = session.execute(
    #     select(CovariateDefn)
    #     .filter(CovariateDefn.name == args.covariate_name)).scalar_one_or_none()

    query = (
        select(CovariateValue.value)
        .join(CovariateDefn)
        .filter(CovariateDefn.name == args.covariate_name)
    ).distinct()
    results = session.execute(query).all()
    print(results)
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
