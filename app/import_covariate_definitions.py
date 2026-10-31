"""Import contributor definitions for named studies and variables into PCLD."""
import argparse
import csv
from pathlib import Path

from sqlalchemy import URL, create_engine, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.pool import NullPool

from database_model_definitions import CovariateDefn, Study, StudyCovariateDefinition


def import_definitions(database, definitions):
    path = Path(database).resolve(strict=True)
    engine = create_engine(URL.create('sqlite', database=str(path)), poolclass=NullPool)
    table = StudyCovariateDefinition.__table__
    with Path(definitions).open(encoding='utf-8-sig', newline='') as stream, engine.begin() as conn:
        reader = csv.DictReader(stream)
        if not {'study_id', 'variable', 'definition', 'units'}.issubset(reader.fieldnames or ()):
            raise ValueError('The CSV needs study_id, variable, definition and units columns.')

        conn.exec_driver_sql('PRAGMA foreign_keys = ON')
        conn.exec_driver_sql('BEGIN IMMEDIATE')
        studies = dict(conn.execute(select(Study.name, Study.id_key)).all())
        variables = dict(conn.execute(select(CovariateDefn.name, CovariateDefn.id_key)).all())
        table.create(conn, checkfirst=True)
        statement = insert(table)
        upsert = statement.on_conflict_do_update(
            index_elements=[table.c.study_id, table.c.covariate_defn_id],
            set_={'definition': statement.excluded.definition, 'units': statement.excluded.units})
        seen = set()
        for row in reader:
            pair = (row['study_id'], row['variable'])
            if pair in seen:
                raise ValueError(f'Duplicate study and variable at CSV line {reader.line_num}: {pair}')
            conn.execute(upsert, {
                'study_id': studies[row['study_id']],
                'covariate_defn_id': variables[row['variable']],
                'definition': row['definition'],
                'units': row['units'],
            })
            seen.add(pair)
    return len(seen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--definitions', required=True, type=Path)
    args = parser.parse_args()
    count = import_definitions(args.database, args.definitions)
    print(f'Imported {count} study-variable definitions.')
