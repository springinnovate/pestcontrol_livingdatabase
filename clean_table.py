"""Fix duplicate misspelled field names."""
import concurrent.futures
import queue
import ftfy
import csv
#from cchardet import detect
from charset_normalizer import detect
import concurrent.futures
import argparse
import os
import re

from matplotlib import colors
from sklearn.cluster import Birch
import collections
import editdistance
import matplotlib.pyplot as plt
import numpy
import pandas


def _generate_scatter_plot(
        table_path, cluster_resolution, clusters, table):
    """Generate scatter plot of clusters."""
    print('generating scatter plot')
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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='search for similar names in space and edit distance')
    parser.add_argument('table_path', help=(
        'path to CSV data table with "long", "lat" and defined text fields'))
    parser.add_argument(
        '--cluster_resolution', required=True, type=float,
        help='cluster distance size to group names to check for edit distance')
    parser.add_argument(
        '--max_edit_distance', default=7, type=int,
        help='max edit distance to check for')
    parser.add_argument(
        '--min_edit_distance', default=1, type=int,
        help='min edit distance to check for')
    parser.add_argument(
        '--single_word_penalty', default=1, type=int,
        help='edit distance for an entire name missing, default 1')
    parser.add_argument(
        '--field_name', default='technician',
        help='field name to scrub, defaults to "technician"')
    args = parser.parse_args()
    encoding_set = collections.defaultdict(int)
    scrubbed_file_path = f'scrubbed_{os.path.basename(args.table_path)}'
    scrubbed_file = open(scrubbed_file_path, 'wb')

    # TODO: matching an edit dist of 7 and 3.0 on cotton technicians gives a lot
    # of SURAGRO

    # This is where the parallel processing happens.

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
                print(f'{last_percent:5.2f}% complete processing of {args.table_path}')
            scrubbed_file.write(line)
    table = pandas.read_csv(
        scrubbed_file_path, encoding='unicode_escape', engine='python')
    return
    table[args.field_name] = table[args.field_name].str.upper()

    X = numpy.ascontiguousarray(table[['long', 'lat']])

    print('build clusters')
    brc = Birch(threshold=args.cluster_resolution, n_clusters=None)
    print(X)
    brc.fit(X)
    clusters = brc.predict(X)
    table['clusters'] = clusters

    _generate_scatter_plot(
        args.table_path, args.cluster_resolution, clusters, table)

    print('create name to edit distance maps for all names')
    name_to_edit_distance = collections.defaultdict(set)
    for cluster in numpy.unique(clusters):
        unique_names = table[
            table['clusters'] == cluster][args.field_name].dropna().unique()
        names_to_process = set(unique_names)
        future_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            print('submit jobs')
            future_list = []
            for a in unique_names:
                future_list.append(
                    executor.submit(
                        _distance_worker, names_to_process.copy(), a,
                        args.single_word_penalty, args.max_edit_distance))
                names_to_process.remove(a)
            print('process results')
            for future in concurrent.futures.as_completed(future_list):
                for (a, b, edit_distance) in future.result():
                    name_to_edit_distance[a].add((edit_distance, b))
                    name_to_edit_distance[b].add((edit_distance, a))
            print('done processing results')

    for max_edit_distance in range(
            args.min_edit_distance, args.max_edit_distance+1):
        edit_distance_table_path = (
            f'candidate_table_{max_edit_distance}_'
            f'{args.cluster_resolution}.csv')
        print(f'generating {edit_distance_table_path}')
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
