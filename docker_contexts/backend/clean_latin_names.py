"""Fix bad encodings and deduplicate field names."""
from joblib import Parallel, delayed
import argparse
from datetime import datetime
import logging
import multiprocessing
import sys

from rapidfuzz import process, fuzz
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


base_names = []


def find_matches(name):
    name_lower = name.lower()
    print(f'processing {name_lower}')
    matches = process.extract(
        name_lower,
        base_names,
        scorer=fuzz.token_sort_ratio,
        limit=5,
        score_cutoff=70
    )
    if matches:
        output_row = [name] + [match[0] for match in matches]
    else:
        output_row = [name, 'COULD NOT FIND MATCH']
    return output_row


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='check latin species name spellings and suggest alternatives if they are spelled wrong.')
    parser.add_argument(
        'latin_covariate_name',
        help='Covarate name that refers to latin species names.')
    args = parser.parse_args()
    session = SessionLocal()

    LOGGER.info(f'processing {args.latin_covariate_name}')

    unique_latin_names = [x[0] for x in (
        session.query(CovariateValue.value)
        .join(CovariateValue.covariate_defn)
        .filter(CovariateDefn.name == args.latin_covariate_name)
        .distinct()
        .all()
    )]

    LOGGER.info(f'there are {len(unique_latin_names)} unique names')
    print(unique_latin_names)
    base_latin_names_df = pd.read_csv('base_data/gbif-phylum-Arthropoda-20241028.csv')

    global base_names
    base_names = base_latin_names_df.iloc[:, 0].astype(str).str.lower().tolist()
    base_names_set = set(base_names)
    unique_names_lower = [name.lower() for name in unique_latin_names]

    unmatched_names = []
    for original_name, lower_name in zip(unique_latin_names, unique_names_lower):
        if lower_name not in base_names_set:
            unmatched_names.append(original_name)
        else:
            LOGGER.info(f'found {lower_name} in the base names set')

    num_cores = multiprocessing.cpu_count()
    output_rows = Parallel(
        n_jobs=num_cores,
        batch_size='auto')(delayed(find_matches)(name) for name in unmatched_names)

    output_df = pd.DataFrame(output_rows)
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_df.to_csv(f'suggested_corrections_{current_datetime}.csv', index=False, header=False)

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
