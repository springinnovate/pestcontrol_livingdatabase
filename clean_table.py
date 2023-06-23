"""Fix duplicate misspelled field names."""
import locale
import argparse
import string
import collections
import concurrent.futures
import csv
import logging
import os
import re
import sys

from charset_normalizer import detect
from matplotlib import colors
from recordlinkage.preprocessing import clean
from sklearn.cluster import Birch
import editdistance
import ftfy
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas
import recordlinkage

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def _generate_scatter_plot(
        table_path, cluster_resolution, clusters, table):
    """Generate scatter plot of clusters."""
    LOGGER.info('generating scatter plot')
    fig, ax = plt.subplots()
    colorlist = list(colors.ColorConverter.colors.keys())
    for i, cluster in enumerate(numpy.unique(clusters)):
        df = table[table['clusters'] == cluster]
        df.plot.scatter(
            x='long', y='lat', ax=ax, s=0.01, marker='x',
            color=colorlist[i % len(colorlist)])
        ax.set_title(
            f'{os.path.basename(table_path)} {len(clusters)} clusters\n'
            f'within ${cluster_resolution}^\\circ$ of each other')
    plt.savefig(
        f'{os.path.basename(os.path.splitext(table_path)[0])}.png', dpi=300)


def _modified_edit_distance(a, b, single_word_penalty):
    """Generate edit distance but account for separate words."""
    # remove all commas
    a_local = re.sub(',', '', a)
    b_local = re.sub(',', '', b)

    # evaluate edit distance if all spaces are removed
    base_distance = editdistance.eval(
        re.sub(' ', '', a_local),
        re.sub(' ', '', b_local))

    # compare edit distance for individual words from smallest to largest
    a_words = [x for x in a_local.split(' ') if x != '']
    b_words = [x for x in b_local.split(' ') if x != '']
    running_edit_distance = 0
    for edit_distance, (x, y) in sorted([
            (editdistance.eval(x, y), (x, y))
            for y in b_words for x in a_words]):
        if x not in a_words or y not in b_words:
            continue
        running_edit_distance += edit_distance
        a_words.remove(x)
        b_words.remove(y)
    running_edit_distance += (len(a_words)+len(b_words))*single_word_penalty
    # smaller of comparing individual words vs single string with no spaces
    return min(base_distance, running_edit_distance)


def _distance_worker(names_to_process, a, single_word_penalty, max_edit_distance):
    result = []
    for b in names_to_process:
        edit_distance = _modified_edit_distance(
            a, b, single_word_penalty)
        if edit_distance > max_edit_distance:
            continue
        result.append((a, b, edit_distance))
    return result


def _process_line(raw_line):
    # guess the encoding
    encoding = detect(raw_line)['encoding']
    line = raw_line.decode(encoding)
    # remove all the in-word quotes
    quoteless = [
        item.replace('"', '')
        for item in next(iter(csv.reader([line])))]
    # run any unicode fixes
    fixed_line = [
        ftfy.fix_text(element).replace('"', '')
        for element in quoteless]
    return (','.join(
        [f'"{val}"' for val in fixed_line])+'\n').encode('utf-8')


def count_valid_characters(name):
    other_valid_letters = ['á', 'õ', 'ě', 'ë', 'ő', 'í', 'í', 'Á']
    return sum(
        1 for char in name if char in list(string.ascii_letters) +
        other_valid_letters)

def attempt_to_correct(original_str):
    if not isinstance(original_str, str):
        return original_str
    original_str = original_str.lower()
    print(original_str)
    if original_str.startswith('mateos salido, josÃ© luis'):
        print(original_str)
        sys.exit()
    encoding_schemes_to_test = ['latin1', 'utf-8']
    artifacts = ['Ã©', 'Ã±', '玡', 'ך', 'ح', ]
    replace_list = [
        (' چ', 'Á'), ('_', 'é'), ('¨', 'Á'), ('oőe', 'oñe'),
        ('äéez', 'óñez')]
    cleaned_str = original_str
    for base, replace in replace_list:
        cleaned_str = cleaned_str.replace(base, replace)

    best_str = cleaned_str
    best_count = count_valid_characters(cleaned_str)

    for encoding_scheme in encoding_schemes_to_test:
        try:
            corrected = cleaned_str.encode(encoding_scheme).decode('utf-8')
            valid_count = count_valid_characters(corrected)
            valid_count -= sum(1 for char in corrected if char in artifacts)
            if valid_count > best_count:
                best_count = valid_count
                best_str = cleaned_str
        except Exception as e:
            continue
    return best_str

def main():
    # print(attempt_to_correct('jos_ maría coronel bejarano 23', 'utf-8'))
    # return
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='search for similar names in space and edit distance')
    parser.add_argument('table_path', help=(
        'path to CSV data table with "long", "lat" and defined text fields'))
    parser.add_argument(
        '--field_name', default='technician',
        help='field name to scrub, defaults to "technician"')
    args = parser.parse_args()
    encoding_set = collections.defaultdict(int)
    scrubbed_file_path = f'scrubbed_{os.path.basename(args.table_path)}'

    # TODO: matching an edit dist of 7 and 3.0 on cotton technicians gives a lot
    # of SURAGRO

    # This is where the parallel processing happens.
    if not (os.path.exists(scrubbed_file_path)):
        scrubbed_file = open(scrubbed_file_path, 'wb')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with open(args.table_path, 'rb') as table_file:
                n_lines = len(['x' for line in table_file])
                table_file.seek(0)
                processed_lines = executor.map(
                    _process_line, [line for line in table_file])
            last_percent = 0
            for line_no, line in enumerate(processed_lines):
                percent_complete = line_no/n_lines*100
                if percent_complete-last_percent >= 1:
                    last_percent = percent_complete
                    LOGGER.info(f'{last_percent:5.2f}% complete processing of {args.table_path}')
                scrubbed_file.write(line)
    table = pandas.read_csv(
        scrubbed_file_path, encoding='utf-8', engine='python')

    LOGGER.info('Create a pandas DataFrame')
    table[args.field_name] = table[args.field_name].apply(attempt_to_correct)
    return
    clean_names = pandas.DataFrame(
        pandas.Series(clean(table[args.field_name]).unique()),
        columns=[args.field_name])

    LOGGER.info('Create an indexer object')
    indexer = recordlinkage.Index()
    indexer.full()

    LOGGER.info('Generate pairs')
    pairs = indexer.index(clean_names)
    LOGGER.info(pairs)
    LOGGER.info('Create a Compare object')
    compare_cl = recordlinkage.Compare()

    LOGGER.info(f'Find exact matches for the {args.field_name} column')
    compare_cl.string(args.field_name, args.field_name, method='lcs')

    LOGGER.info('Compute the comparison')
    features = compare_cl.compute(pairs, clean_names)

    LOGGER.info('Find potential matches')
    print(features)
    potential_matches = features[features.sum(axis=1) > 0.8]  # Adjust the threshold as needed

    LOGGER.info('Create a graph to store the matches')
    G = nx.Graph()

    LOGGER.info('Add an edge for each match')
    for match in potential_matches.index:
        G.add_edge(match[0], match[1])

    LOGGER.info('Find connected components, which correspond to sets of matches')
    match_sets = list(nx.connected_components(G))

    candidate_table = f'candidate_table_{args.field_name}.csv'
    LOGGER.info(f'Generating {candidate_table}')
    processed_set = set()

    with open(candidate_table, 'wb') as candidate_table:
        candidate_table.write(b'\xEF\xBB\xBF')
        for match_set in match_sets:
            similar_list = [clean_names.loc[i, args.field_name] for i in match_set]
            # Sort names by length (descending) and number of non-ASCII characters (ascending)
            similar_list = list(sorted(
                similar_list, key=lambda name: -count_valid_characters(name)))

            corrected = attempt_to_correct(similar_list[0])
            if corrected != similar_list[0]:
                similar_list.insert(0, corrected)
            LOGGER.info(similar_list)

            candidate_table.write(
                (','.join([f'"{name}"' for name in similar_list]) + '\n').encode('utf-8'))
            local_table = table.copy()
            local_table.replace(
                {args.field_name: similar_list[1:]}, similar_list[0],
                inplace=True)
            local_table[args.field_name] = clean(local_table[args.field_name])
            local_table.to_csv(args.field_name + '.csv')
    return

    table[args.field_name] = table[args.field_name].str.upper()
    X = numpy.ascontiguousarray(table[['long', 'lat']])

    LOGGER.info('build clusters')
    brc = Birch(threshold=args.cluster_resolution, n_clusters=None)
    LOGGER.info(X)
    brc.fit(X)
    clusters = brc.predict(X)
    table['clusters'] = clusters

    _generate_scatter_plot(
        args.table_path, args.cluster_resolution, clusters, table)

    LOGGER.info('create name to edit distance maps for all names')
    name_to_edit_distance = collections.defaultdict(set)
    for cluster in numpy.unique(clusters):
        unique_names = table[
            table['clusters'] == cluster][args.field_name].dropna().unique()
        names_to_process = set(unique_names)
        future_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            LOGGER.info('submit jobs')
            future_list = []
            for a in unique_names:
                future_list.append(
                    executor.submit(
                        _distance_worker, names_to_process.copy(), a,
                        args.single_word_penalty, args.max_edit_distance))
                names_to_process.remove(a)
            LOGGER.info('process results')
            for future in concurrent.futures.as_completed(future_list):
                for (a, b, edit_distance) in future.result():
                    name_to_edit_distance[a].add((edit_distance, b))
                    name_to_edit_distance[b].add((edit_distance, a))
            LOGGER.info('done processing results')

    for max_edit_distance in range(
            args.min_edit_distance, args.max_edit_distance+1):
        edit_distance_table_path = (
            f'candidate_table_{max_edit_distance}_'
            f'{args.cluster_resolution}.csv')
        LOGGER.info(f'generating {edit_distance_table_path}')
        processed_set = set()
        with open(edit_distance_table_path, 'w', encoding="ISO-8859-1") as \
                candidate_table:
            local_table = table.copy()
            for base_name, edit_distance_set in name_to_edit_distance.items():
                if base_name in processed_set:
                    continue
                replace_name_list = []
                processed_set.add(base_name)
                row = f'"{base_name}"'
                for edit_distance, name in sorted(edit_distance_set):
                    if name == base_name:
                        continue
                    if edit_distance > max_edit_distance:
                        break
                    replace_name_list.append(name)
                    processed_set.add(name)
                    row += f',"{name}"'
                if replace_name_list is not []:
                    local_table.replace(
                        {args.field_name: replace_name_list}, base_name,
                        inplace=True)
                candidate_table.write(f'{row}\n')
        target_table_path = (
            f'remapped_{max_edit_distance}_'
            f'{args.cluster_resolution}_{os.path.basename(args.table_path)}')
        local_table.to_csv(target_table_path, index=False)


if __name__ == '__main__':
    main()
