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
import pandas as pd
import recordlinkage

from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from database_model_definitions import CovariateValue, CovariateDefn


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


def count_valid_characters(name):
    other_valid_letters = ['á', 'õ', 'ě', 'ë', 'ő', 'í', 'í', 'Á']
    return sum(
        1 for char in name if char in list(string.ascii_letters) +
        other_valid_letters)


def _train_classifier():
    table = pd.read_csv('modified_training.csv')
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
        description='search for similar spellings in database covariates.')
    parser.add_argument('covariate_name', nargs='+', help='Covarate name to search values through.')
    args = parser.parse_args()
    session = SessionLocal()
    classifier = _train_classifier()

    # now iterate through the args.field_name pairs ....
    for covariate_name in args.covariate_name:
        LOGGER.info(f'processing {covariate_name}')

        unique_covariate_values = (
            session.query(CovariateValue.value)
            .join(CovariateValue.covariate_defn)
            .filter(CovariateDefn.name == covariate_name)
            .distinct()
            .all()
        )

        clean_names = pd.DataFrame(
            unique_covariate_values,
            columns=[covariate_name])

        indexer = recordlinkage.Index()
        indexer.full()

        pairs = indexer.index(clean_names)
        compare_cl = recordlinkage.Compare(n_jobs=-1)

        for method in [
                'qgram', 'cosine', 'smith_waterman', 'lcs']:
            compare_cl.string(covariate_name, covariate_name, method=method)
        LOGGER.info(f'Compute the comparison: {pairs}\n\n{clean_names}')
        try:
            features = compare_cl.compute(pairs, clean_names, clean_names)
            table_columns = {
                'probability': [],
                'str1': [],
                'str2': [],
            }
            prob_array_list = []
            for prob_array, index_array in zip(features.values, features.index):
                str1 = clean_names.loc[index_array[0], covariate_name]
                str2 = clean_names.loc[index_array[1], covariate_name]
                table_columns['str1'].append(str1)
                table_columns['str2'].append(str2)
                prob_array_list.append(
                    numpy.append(prob_array, [len(str1)/MAX_LINE_LEN, len(str2)/MAX_LINE_LEN]))
            LOGGER.info('running prediction')
            result = classifier.predict(prob_array_list)
            table_columns['probability'] = result
            df = pd.DataFrame(table_columns)
            df = df.sort_values(by='probability', ascending=False)
            df.to_csv(f'{covariate_name}.csv', index=False)
        except AttributeError:
            LOGGER.warn(f'too few pairs to compare on {covariate_name}, no duplicates probably')

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
