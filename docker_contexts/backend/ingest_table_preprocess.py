"""Script to take in a CSV table and put it in the database."""
import argparse
import glob
from itertools import zip_longest

from concurrent.futures import ProcessPoolExecutor
from database import SessionLocal, init_db
from database_model_definitions import Study, DOI, Sample, Covariate
from sqlalchemy import inspect
import pandas

TABLE_MAPPING_PATH = 'table_mapping.csv'


def main():
    init_db()
    db = SessionLocal()
    parser = argparse.ArgumentParser(description='parse table')
    parser.add_argument('sample_table_path', help='Path to sample table')
    args = parser.parse_args()

    df = pandas.read_csv(args.sample_table_path)
    print(df.columns)

    inspector = inspect(Study)
    study_columns = [(column.name, column.nullable) for column in inspector.columns]
    inspector = inspect(Sample)
    sample_columns = [(column.name, column.nullable) for column in inspector.columns]
    inspector = inspect(Covariate)
    covariate_columns = [(column.name, column.nullable) for column in inspector.columns]

    with open('table_mapping.csv', 'w') as table:
        table.write('input columns,database column,base columns,required\n')
        for column_val, csv_column_name in zip_longest(
                study_columns+sample_columns+covariate_columns,
                df.columns
                ):
            if column_val is None:
                column_name, optional_column = '', ''
            else:
                column_name, optional_column = column_val
            if column_name == 'id_key':
                continue
            table.write(f'{csv_column_name},,{column_name},{optional_column}\n')

    return
    with ProcessPoolExecutor() as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_pdf, file_path)
            future_list.append(future)
        article_list = [
            article for future in future_list for article in future.result()]
    upsert_articles(db, article_list)
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
