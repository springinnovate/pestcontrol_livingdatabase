"""Scrub table fields."""
import pandas
import editdistance
TABLE_PATH = 'plotdata.cotton.select.csv'


def main():
    """Entry point."""
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
