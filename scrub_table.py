"""Replace a column with user defined table values."""
import argparse
import collections
import csv
import logging
import os
import sys

from recordlinkage.preprocessing import clean
import networkx as nx
import pandas

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=(
        'replace column in CSV with replacement values in a second table.'))
    parser.add_argument('table_path', help='Path to table.')
    args = parser.parse_args()
    base_table = pandas.read_csv(
        args.table_path, encoding='utf-8', engine='python')
    base_table = clean(base_table)
    base_table.to_csv('clean.csv')


if __name__ == '__main__':
    main()
