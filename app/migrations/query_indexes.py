"""Back up a PCLD SQLite database, add two query indexes, and run full ANALYZE.

This is a manual migration. Run against a copy first. Every run requires fresh
backup/report paths and refreshes statistics for all indexes, even if the two
indexes already exist. Application rows and existing indexes are unchanged.
The script never imports app.database or starts application services.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


INDEX_DEFINITIONS = {
    'idx_covariate_value_defn_value_sample': (
        'covariate_value', ('covariate_defn_id', 'value', 'sample_id')),
    'idx_sample_study_point': ('sample', ('study_id', 'point_id')),
}


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def schema(connection):
    return connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()


def statistics(connection):
    if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'").fetchone():
        return []
    return connection.execute('SELECT tbl,idx,stat FROM sqlite_stat1 ORDER BY tbl,idx').fetchall()


def validate_indexes(connection):
    """Return already-present targets; reject any conflicting named object."""
    existing = []
    for name, (table, columns) in INDEX_DEFINITIONS.items():
        table_columns = {row[1] for row in connection.execute(
            f'PRAGMA table_info({quote_identifier(table)})')}
        if not set(columns) <= table_columns:
            raise ValueError(f'Missing required columns on {table}: {columns}')
        objects = connection.execute(
            'SELECT type,tbl_name FROM sqlite_master WHERE name=? COLLATE NOCASE',
            (name,)).fetchall()
        if not objects:
            continue
        if objects != [('index', table)]:
            raise ValueError(f'Conflicting schema object named {name}')
        index_row = next(row for row in connection.execute(
            f'PRAGMA index_list({quote_identifier(table)})') if row[1].lower() == name.lower())
        # index_list: seq, name, unique, origin, partial.
        key_rows = [row for row in connection.execute(
            f'PRAGMA index_xinfo({quote_identifier(name)})') if row[5]]
        if (index_row[2] or index_row[3] != 'c' or index_row[4]
                or tuple(row[2] for row in key_rows) != columns
                or any(row[3] != 0 or row[4] != 'BINARY' for row in key_rows)):
            raise ValueError(f'Conflicting index definition for {name}')
        existing.append(name)
    return existing


def check_integrity(connection):
    result = connection.execute('PRAGMA quick_check').fetchall()
    if result != [('ok',)]:
        raise RuntimeError(f'SQLite quick_check failed: {result[:10]}')
    foreign_keys = connection.execute('PRAGMA foreign_key_check').fetchmany(10)
    if foreign_keys:
        raise RuntimeError(f'Foreign key check failed: {foreign_keys}')


def save_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def migrate(database_path, backup_path, report_path):
    database_path = Path(database_path).resolve(strict=True)
    backup_path = Path(backup_path).resolve()
    report_path = Path(report_path).resolve()
    if len({database_path, backup_path, report_path}) != 3:
        raise ValueError('Database, backup and report must be different paths')
    for path in (backup_path, report_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing file: {path}')

    report = {
        'database': str(database_path), 'backup': None,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'sqlite_version': sqlite3.sqlite_version, 'status': 'preparing',
        'indexes_created': [], 'indexes_reused': [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    connection = None
    committed = False
    try:
        connection = sqlite3.connect(database_path.as_uri() + '?mode=rw', uri=True, timeout=30)
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        existing = validate_indexes(connection)
        before_schema = schema(connection)
        report.update(indexes_reused=existing, statistics_before=statistics(connection))

        # The reserved writer lock prevents changes between the backup and the
        # migration. Back up through a separate reader rather than the active
        # write connection, which would wait on its own transaction.
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open('xb'):
            pass
        print(f'Backing up {database_path} to {backup_path}', flush=True)
        with closing(sqlite3.connect(database_path.as_uri() + '?mode=ro', uri=True)) as reader, \
                closing(sqlite3.connect(backup_path)) as destination:
            reader.execute('PRAGMA query_only=ON')
            reader.backup(destination)
            check_integrity(destination)
        report['backup'] = str(backup_path)
        save_report(report_path, report)

        for name, (table, columns) in INDEX_DEFINITIONS.items():
            if name in existing:
                continue
            statement = (
                f'CREATE INDEX {quote_identifier(name)} ON {quote_identifier(table)}'
                f'({", ".join(quote_identifier(column) for column in columns)})')
            print(statement, flush=True)
            connection.execute(statement)
            report['indexes_created'].append(name)

        # Refresh all statistics, not just those of the new indexes. In measured
        # PCLD queries, old statistics prevented selection of the new indexes.
        print('Refreshing all SQLite query statistics with ANALYZE', flush=True)
        connection.execute('ANALYZE')
        validate_indexes(connection)
        after_schema = schema(connection)
        unchanged = [row for row in after_schema if row[1] not in report['indexes_created']]
        if unchanged != before_schema:
            raise RuntimeError('Unexpected change to an existing schema object')
        expected_new = {('index', name) for name in report['indexes_created']}
        actual_new = {(row[0], row[1]) for row in after_schema if row not in before_schema}
        if actual_new != expected_new:
            raise RuntimeError('Unexpected additional schema objects')
        check_integrity(connection)
        report.update(
            statistics_after=statistics(connection), analyze='full',
            existing_schema_preserved=True, application_row_writes=0,
            quick_check='ok', foreign_key_check='ok')
        connection.commit()
        committed = True
        report.update(status='migrated', completed_utc=datetime.now(timezone.utc).isoformat())
        save_report(report_path, report)
        return report
    except Exception as error:
        if committed:
            report.update(status='migrated_report_error', report_error=str(error))
            try:
                save_report(report_path, report)
            except OSError:
                pass
            return report
        if connection and connection.in_transaction:
            connection.rollback()
        report.update(status='failed', error=str(error))
        try:
            save_report(report_path, report)
        except OSError:
            pass
        raise
    finally:
        if connection:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.database, args.backup, args.report), indent=2))


if __name__ == '__main__':
    main()
