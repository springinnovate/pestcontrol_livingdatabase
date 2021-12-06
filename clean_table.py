"""Scrub table fields."""
import os
import threading

from matplotlib import colors
from sklearn.cluster import Birch
import collections
import editdistance
import matplotlib.pyplot as plt
import numpy
import pandas

TABLE_PATH = 'plotdata.cotton.select.csv'
CLUSTER_RESOLUTION = 0.01


def _generate_scatter_plot(clusters, table):
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
            f'{os.path.basename(TABLE_PATH)} {len(clusters)} clusters\n'
            f'within ${CLUSTER_RESOLUTION}^\\circ$ of each other')
    plt.savefig(
        f'{os.path.basename(os.path.splitext(TABLE_PATH)[0])}.png', dpi=300)


def _modified_edit_distance(a, b):
    """Generate edit distance but account for separate words."""
    # drop commas
    a_local = a.lower().replace(',', '')
    b_local = b.lower().replace(',', '')
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
    print(a_words, b_words)
    running_edit_distance += len(a_words)+len(b_words)
    return running_edit_distance


def main():
    """Entry point."""
    table = pandas.read_csv(
        TABLE_PATH, encoding='unicode_escape', engine='python')
    X = table[['long', 'lat']]
    print(X['long'].min(), X['long'].max())

    brc = Birch(threshold=0.01, n_clusters=None)
    brc.fit(X.values)
    clusters = brc.predict(X.values)
    table['clusters'] = clusters

    _generate_scatter_plot(clusters, table)

    name_to_edit_distance = collections.defaultdict(set)
    for cluster in numpy.unique(clusters):
        unique_names = table[
            table['clusters'] == cluster]['technician'].dropna().unique()

        # process this subset of names
        # first filter by names we've already identified
        [name_to_edit_distance[a].update([(_modified_edit_distance.eval(a, b), b)
         for b in unique_names])
         for a in unique_names]

    for max_edit_distance in range(1, 8):
        edit_distance_table_path = f'candidate_table_{max_edit_distance}.csv'
        print(f'generating {edit_distance_table_path}')
        processed_set = set()
        with open(edit_distance_table_path, 'w', encoding="ISO-8859-1") as \
                candidate_table:
            for base_name, edit_distance_set in name_to_edit_distance.items():
                if base_name in processed_set:
                    continue
                processed_set.add(base_name)
                row = f'"{base_name}"'
                for edit_distance, name in sorted(edit_distance_set):
                    if name == base_name:
                        continue
                    if edit_distance > max_edit_distance:
                        break
                    processed_set.add(name)
                    row += f',"{name}"'
                candidate_table.write(f'{row}\n')


if __name__ == '__main__':
    main()
