"""Scrub table fields."""
import numpy
import pandas
import editdistance
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from matplotlib import colors

TABLE_PATH = 'plotdata.cotton.select.csv'


def main():
    """Entry point."""
    table = pandas.read_csv(
        TABLE_PATH, encoding='unicode_escape', engine='python')
    print(TABLE_PATH)
    print(table.columns)
    X = table[['long', 'lat']]
    print(X)
    fig, ax = plt.subplots()
    brc = Birch(threshold=0.01, n_clusters=None)
    brc.fit(X)
    groups = brc.predict(X)
    print(len(X))
    print(len(groups))
    X['groups'] = groups
    print(len(X))
    technicians = table['technician']
    print(len(technicians))
    print(len(technicians))
    processed_names = set()
    for group in numpy.unique(groups):
        unique_tech = table[X['groups']==group]['technician'].unique()
        print(f'{group}: {unique_tech}')

        # process this subset of names
        # first filter by names we've already identified
        local_name_set = set(unique_tech)-processed_names
        cross_product = {
            a: sorted(
                [(editdistance.eval(a, b), b)
                 for b in local_name_set if a != b])
            for a in local_name_set}

        for name in local_name_set:
            for edit_distance, candidate in cross_product[name]:
                if candidate not in name_set:
                    continue
                if edit_distance > max_edit_distance:
                    break
                row += f',"{candidate}"'
                name_set.remove(candidate)

    for max_edit_distance in range(1, 5):
        with open(f'candidate_table_{max_edit_distance}.csv', 'w', encoding="utf-8") as candidate_table:
            for name, count in zip(names_by_count.index, names_by_count):
                print(name, count)
                if name not in name_set:
                    continue
                name_set.remove(name)
                row = f'"{name}"'
                for edit_distance, candidate in cross_product[name]:
                    if candidate not in name_set:
                        continue
                    if edit_distance > max_edit_distance:
                        break
                    row += f',"{candidate}"'
                    name_set.remove(candidate)
                candidate_table.write(f'{row}\n')

    return

    print(f'{len(X)} groups')
    #table.plot.scatter(y='lat', x='long', s=0.5)
    colorlist = list(colors.ColorConverter.colors.keys())
    for i, group in enumerate(numpy.unique(groups)):
        print(group)
        df = X[X['groups'] == group]
        df.plot.scatter(
            x='long', y='lat', ax=ax, s=0.2,
            color=colorlist[i % len(colorlist)])
    plt.savefig('plot.png', dpi=600)
    print(numpy.unique(groups))
    return



    for max_edit_distance in range(1, 10):
        print(f'max edit edit_distance {max_edit_distance}')
        table = pandas.read_csv(
            TABLE_PATH, encoding='unicode_escape', engine='python')
        all_names_df = table['region'].dropna().apply(lambda x: x.lower())
        names_by_count = all_names_df.value_counts()

        name_set = set(names_by_count.index)
        cross_product = {
            a: sorted(
                [(editdistance.eval(a, b), b) for b in name_set if a != b])
            for a in name_set}

        with open(f'candidate_table_{max_edit_distance}.csv', 'w', encoding="utf-8") as candidate_table:
            for name, count in zip(names_by_count.index, names_by_count):
                print(name, count)
                if name not in name_set:
                    continue
                name_set.remove(name)
                row = f'"{name}"'
                for edit_distance, candidate in cross_product[name]:
                    if candidate not in name_set:
                        continue
                    if edit_distance > max_edit_distance:
                        break
                    row += f',"{candidate}"'
                    name_set.remove(candidate)
                candidate_table.write(f'{row}\n')


if __name__ == '__main__':
    main()
