"""Replace a column with user defined table values."""
import io
import codecs
import csv
import chardet
import argparse
import os
from recordlinkage.preprocessing import clean

import pandas


def clean_io(path):
    with open(path, 'rb') as f:
        bytes_content = f.read()
    encoding_result = chardet.detect(bytes_content)

    print(f'{path} -- {encoding_result}')
    encoding = encoding_result['encoding']
    print(encoding)
    if encoding is None or encoding_result['confidence'] < 0.9:
        encoding = 'utf-8'
    # replacing erroneous characters with � (U+FFFD, the official Unicode replacement character)
    content = bytes_content.decode(encoding, errors='replace')
    data = io.StringIO(content)
    return data


def _clean(series):
    print(series.decode('utf-8'))
    return clean(series).decode('utf-8')


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
    base_table = pandas.read_csv(clean_io(args.base_table_path))

    # Select only the string columns, then apply the function
    base_table.loc[:, base_table.dtypes == object] = base_table.select_dtypes(
        include=[object]).apply(clean)

    print(base_table['technician'])
    replacement_table = csv.reader(
        clean_io(args.replacement_table_path))
    for row in replacement_table:
        print(row)
        print(base_table[args.field_name])
        if len(row) <= 1:
            continue
        base_table.replace(
           {args.field_name: row[1:]}, row[0], inplace=True)

    suffix = args.target_suffix
    if suffix is None:
        suffix = args.field_name
    target_table_path = f'%s{suffix}%s' % os.path.splitext(
        args.base_table_path)
    base_table.to_csv(target_table_path, index=False, encoding='utf-8')
    print(f'result written to {target_table_path}')


if __name__ == '__main__':
    main()
