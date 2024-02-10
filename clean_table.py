"""Fix bad encodings and deduplicate field names."""
import argparse
import string
import concurrent.futures
import csv
import logging
import os
import re
import sys

from matplotlib import colors
from recordlinkage.preprocessing import clean
from sklearn.linear_model import LinearRegression
import editdistance
import ftfy
import matplotlib.pyplot as plt
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

RAW_LOOKUP = [
    ['a_ina', 'acina'],
    ['jos_', 'jose'],
    ['a_a', 'ana'],
    ['a_e', 'ane'],
    ['a_i', 'ani'],
    ['a_l', 'ael'],
    ['a_n', 'aen'],
    ['a_o', 'ano'],
    ['e_a', 'ena'],
    ['e_o', 'eno'],
    ['g_n', 'gen'],
    ['i_a', 'ina'],
    ['i_e', 'ine'],
    ['i_n', 'ien'],
    ['i_r', 'ier'],
    ['i_u', 'inu'],
    ['m_n', 'men'],
    ['o_a', 'ona'],
    ['p_r', 'per'],
    ['u_a', 'una'],
    ['u_e', 'nue'],
    ['u_o', 'uno'],
    ['d_e', 'done'],
    ['r_s', 'ros'],
    ['n_', 'na'],
    ['m_', 'ma'],
    ['o_e', 'one'],
    ]
MAX_LINE_LEN = 27

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


def _replace_common_substring_errors(base_string):
    for substring, replacement in RAW_LOOKUP:
        base_string = base_string.replace(substring, replacement)
    return base_string


def _process_line(raw_line):
    line = raw_line.decode('utf-8', errors='ignore')
    # remove all the in-word quotes
    quoteless = [
        item.replace('"', '')
        for item in next(iter(csv.reader([line])))]
    # run any unicode fixes
    fixed_line = [
        # _replace_common_substring_errors(
        #     ftfy.fix_text(
        #         element, normalization='NFKC').replace('"', '').lower())
        ftfy.fix_text(element, normalization='NFKC').replace('"', '')
        for element in quoteless]
    return (','.join(
        [f'"{val}"' for val in fixed_line])+'\n').encode('utf-8')


def count_valid_characters(name):
    other_valid_letters = ['á', 'õ', 'ě', 'ë', 'ő', 'í', 'í', 'Á']
    return sum(
        1 for char in name if char in list(string.ascii_letters) +
        other_valid_letters)


def _train_classifier():
    table = pandas.read_csv('modified_training.csv')
    X_set = table[['qgram', 'cosine', 'smith_waterman', 'lcs', 'len_a', 'len_b']]
    y_vector = table['category']
    reg = LinearRegression().fit(X_set, y_vector)
    reg.fit(X_set, y_vector)
    return reg


def main():
    # print(attempt_to_correct('jos_ maría coronel bejarano 23', 'utf-8'))
    # return
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='search for similar names in space and edit distance')
    parser.add_argument('table_path', help=(
        'path to CSV data table with "long", "lat" and defined text fields'))
    parser.add_argument(
        '--field_name', nargs='+', required=True,
        help='a list of field names in the table to deduplicate.')
    args = parser.parse_args()
    scrubbed_file_path = f'scrubbed_{os.path.basename(args.table_path)}'

    classifier = _train_classifier()

    if not os.path.exists(scrubbed_file_path):
        scrubbed_file = open(scrubbed_file_path, 'wb')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with open(args.table_path, 'rb') as table_file:
                n_lines = len(['x' for line in table_file])
                table_file.seek(0)
                processed_lines = executor.map(
                    _process_line, list(table_file))
            last_percent = 0
            scrubbed_file.write(b'\xEF\xBB\xBF')
            missing_letter_set = set()
            for line_no, line in enumerate(processed_lines):
                missing_letter_set.update([
                    word.replace('"', '')
                    for element in line.decode('utf-8').split(',') if '_' in element
                    for word in element.split(' ')
                    if '_' in word])
                percent_complete = line_no/n_lines*100
                if percent_complete-last_percent >= 1:
                    last_percent = percent_complete
                    LOGGER.info(f'{last_percent:5.2f}% complete processing of {args.table_path}')
                scrubbed_file.write(line)
        with open('missing.txt', 'w') as file:
            file.write('\n'.join(sorted(missing_letter_set)))
    table = None
    for encoding in ['utf8', 'latin1', 'cp1252']:
        try:
            table = pandas.read_csv(args.table_path, engine='python', encoding=encoding)
            break
        except UnicodeDecodeError:
            pass
    if table is None:
        raise RuntimeError(f'could not decode {args.table_path}')
    LOGGER.info(f'{table.head()}')
    LOGGER.info(f'{table.columns}')

    # now iterate through the args.field_name pairs ....
    prob_array_list = []
    match_pair_list = []
    for field_name in args.field_name:
        LOGGER.info(f'processing {field_name}')
        clean_names = pandas.DataFrame(
            pandas.Series(clean(table[field_name]).dropna().unique()),
            columns=[field_name])

        indexer = recordlinkage.Index()
        indexer.full()

        pairs = indexer.index(clean_names)
        compare_cl = recordlinkage.Compare(n_jobs=-1)

        for method in [
                'qgram', 'cosine', 'smith_waterman', 'lcs']:
            compare_cl.string(field_name, field_name, method=method)

        LOGGER.info(f'Compute the comparison: {pairs}\n\n{clean_names}')
        try:
            features = compare_cl.compute(pairs, clean_names, clean_names)

            for prob_array, index_array in zip(features.values, features.index):
                val1 = clean_names.loc[index_array[0], field_name]
                val2 = clean_names.loc[index_array[1], field_name]
                match_pair_list.append((val1, val2, field_name))
                prob_array_list.append(
                    numpy.append(prob_array, [len(val1)/MAX_LINE_LEN, len(val2)/MAX_LINE_LEN]))
        except AttributeError:
            LOGGER.warn(f'too few pairs to compare on {field_name}, no duplicates probably')

    classification_list = classifier.predict(prob_array_list)
    with open(f'replacement_{scrubbed_file_path}', 'w') as file:
        file.write(f'{scrubbed_file_path}\n')
        for raw_class, match_pair in sorted(zip(classification_list, match_pair_list), reverse=True):
            if raw_class < 1:
                continue
            file.write(f'{int(numpy.round(raw_class+0.25))},{",".join(match_pair)}\n')


if __name__ == '__main__':
    main()
