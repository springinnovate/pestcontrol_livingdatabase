"""Scrub table fields."""
import argparse
import os

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
    # drop commas
    a_local = a.replace(',', '')
    b_local = b.replace(',', '')
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
    return running_edit_distance


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='search for similar names in space and edit distance')
    parser.add_argument('table_path', help=(
        'path to CSV data table with "long", "lat" and "technician" fields'))
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
    args = parser.parse_args()
    table = pandas.read_csv(
        args.table_path, encoding='unicode_escape', engine='python')

    table['technician'] = table['technician'].str.upper()
    print(table['technician'])

    X = table[['long', 'lat']]
    print(X['long'].min(), X['long'].max())

    brc = Birch(threshold=args.cluster_resolution, n_clusters=None)
    brc.fit(X.values)
    clusters = brc.predict(X.values)
    table['clusters'] = clusters

    _generate_scatter_plot(
        args.table_path, args.cluster_resolution, clusters, table)

    name_to_edit_distance = collections.defaultdict(set)
    for cluster in numpy.unique(clusters):
        unique_names = table[
            table['clusters'] == cluster]['technician'].dropna().unique()

        # process this subset of names
        # first filter by names we've already identified
        [name_to_edit_distance[a].update(
            [(_modified_edit_distance(a, b, args.single_word_penalty), b)
             for b in unique_names])
         for a in unique_names]

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
                        {'technician': replace_name_list}, base_name,
                        inplace=True)
                candidate_table.write(f'{row}\n')
        target_table_path = (
            f'remapped_{max_edit_distance}_'
            f'{args.cluster_resolution}_{os.path.basename(args.table_path)}')
        local_table.to_csv(target_table_path, index=False)


if __name__ == '__main__':
    main()
