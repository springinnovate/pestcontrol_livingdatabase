"""Fix bad encodings and deduplicate field names."""
import argparse
import string
import logging
import re
import sys

from rapidfuzz import process, fuzz
from sklearn.linear_model import LinearRegression
import editdistance
import pandas as pd

from database import SessionLocal
from database_model_definitions import CovariateValue, CovariateDefn


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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
        description='check latin species name spellings and suggest alternatives if they are spelled wrong.')
    parser.add_argument('latin_covariate_name', help='Covarate name that refers to latin species names.')
    args = parser.parse_args()
    session = SessionLocal()

    LOGGER.info(f'processing {args.latin_covariate_name}')

    unique_latin_names = (
        session.query(CovariateValue.value)
        .join(CovariateValue.covariate_defn)
        .filter(CovariateDefn.name == args.latin_covariate_name)
        .distinct()
        .all()
    )

    unique_latin_names = pd.DataFrame(
        unique_latin_names,
        columns=[args.latin_covariate_name])

    base_latin_names_df = pd.read_csv('base_data/all_species_names.csv')

    base_names = base_latin_names_df.iloc[:, 0].astype(str).str.lower().tolist()
    base_names_set = set(base_names)
    unique_names_lower = [name.lower() for name in unique_latin_names]

    unmatched_names = []
    for original_name, lower_name in zip(unique_latin_names, unique_names_lower):
        if lower_name not in base_names_set:
            unmatched_names.append(original_name)

    output_rows = []
    for name in unmatched_names:
        LOGGER.info(f'processing {name}')
        name_lower = name.lower()
        # Find top 5 matches with a minimum similarity score of 80
        matches = process.extract(
            name_lower,
            base_names,
            scorer=fuzz.token_sort_ratio,
            limit=5,
            score_cutoff=80
        )
        # Retrieve original casing for matches
        matches_original_casing = base_latin_names_df.iloc[[base_names.index(match[0]) for match in matches], 0].tolist()
        output_row = [name] + matches_original_casing
        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv('suggested_corrections.csv', index=False, header=False)

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
