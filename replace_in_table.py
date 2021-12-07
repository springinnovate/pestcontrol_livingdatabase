"""Replace a column with user defined table values."""
import csv
import argparse
import os

import pandas


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=(
        'replace column in CSV with replacement values in a second table.'))
    parser.add_argument('base_table_path', help=(
        'path to arbitrary CSV data table'))
    parser.add_argument('replacement_table_path', help=(
        'path to CSV data table containing rows of values to replace, '
        'each row defines strings in the 2 column and beyond to be replaced '
        'with the value in the first column'))
    parser.add_argument(
        '--field_name', required=True, help=(
            'field name to replace in the base_table_path with the values in '
            'the replaement table.'))
    parser.add_argument(
        '--target_suffix', help=(
            'suffix the target table with this value, default is the '
            'replacement fieldname'))
    args = parser.parse_args()
    base_table = pandas.read_csv(
        args.base_table_path, encoding='unicode_escape', engine='python')

    with open(args.replacement_table_path) as replacement_table_file:
        replacement_table = csv.reader(
            replacement_table_file)
        for row in replacement_table:
            if len(row) <= 1:
                continue
            base_table.replace(
               {args.field_name: row[1:]}, row[0], inplace=True)

    suffix = args.target_suffix
    if suffix is None:
        suffix = args.field_name
    target_table_path = f'%s{suffix}%s' % os.path.splitext(
        args.base_table_path)
    base_table.to_csv(target_table_path, index=False)


if __name__ == '__main__':
    main()
