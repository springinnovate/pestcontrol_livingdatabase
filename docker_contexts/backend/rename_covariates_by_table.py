"""Used to rename elemenets in database given the value of the first covariate"""
import argparse
import csv

import pandas as pd
from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, update


def load_corrections_from_csv(file_path):
    df = pd.read_csv(file_path)
    corrections = list(zip(df['Misspelled'], df['Correct spelling']))
    single_word_corrections = [
        (x.split(' ')[0], y.split(' ')[0]) for x, y in corrections]
    return single_word_corrections


def update_covariate_values(session, corrections):
    for misspelled, correct in corrections:
        stmt = (
            update(CovariateValue)
            .where(
                CovariateValue.value == misspelled,
                CovariateValue.covariate_defn.has(CovariateDefn.name == 'Genus')
            )
            .values(value=correct)
        )
        result = session.execute(stmt)
        if result.rowcount > 0:
            print(f"Updated {result.rowcount} entries: '{misspelled}' -> '{correct}'")

    # Commit the changes
    session.commit()
    print("Updates completed successfully.")


def main():
    init_db()
    parser = argparse.ArgumentParser(description='rename or add covariates by table defn')
    parser.add_argument('table_path')
    args = parser.parse_args()

    corrections_list = load_corrections_from_csv(args.table_path)
    session = SessionLocal()
    update_covariate_values(session, corrections_list)
    session.close()


if __name__ == '__main__':
    main()
