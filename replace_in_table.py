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
    parser.add_argument('replacement_table_path', help=(
        'path to a replacement table made by `clean_table.py`'))
    args = parser.parse_args()
    graph_by_fieldname = collections.defaultdict(lambda: nx.Graph())
    with open(args.replacement_table_path, encoding='utf-8') as table_file:
        base_table_path = table_file.readline().rstrip(',\n ')
        LOGGER.debug(base_table_path)
        LOGGER.info('Create a graph to store the matches')

        LOGGER.info('Add an edge for each match')
        for line in table_file:
            classification, str_a, str_b, fieldname = line.rstrip().split(',')
            if classification != '3':
                continue
            graph_by_fieldname[fieldname].add_edge(str_a, str_b)

    base_table = pandas.read_csv(base_table_path)
    for fieldname, graph in graph_by_fieldname.items():
        LOGGER.info(f'fixing {fieldname}')
        connected_components = nx.connected_components(graph)
        for edge_set in connected_components:
            match_list = list(sorted(edge_set, key=lambda x: len(x)))
            LOGGER.debug(match_list)
            for element in match_list[1:]:
                base_table[fieldname] = clean(base_table[fieldname])
                base_table[fieldname] = base_table[fieldname].replace(element, match_list[0])
    target_table_path = f'fixed_{os.path.basename(base_table_path)}'
    LOGGER.info(f'fixed table at {target_table_path}')
    base_table.to_csv(target_table_path)


if __name__ == '__main__':
    main()
