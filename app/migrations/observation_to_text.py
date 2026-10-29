"""Back up a PCLD SQLite database and migrate sample.observation to TEXT.

Uses SQLite's transactional table-rebuild procedure:
https://www.sqlite.org/lang_altertable.html#otheralter
No observations are added, removed, rounded, or interpreted. Python str(float)
preserves the stored float's round-trip value, unlike SQLite's shorter CAST text.
Run against a copy first. This script never imports app.database.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


COLUMNS = ('id_key', 'observation', 'point_id', 'study_id', 'year')
NEW_TABLE = 'sample_observation_text_new'


def observation_text(value):
    if not isinstance(value, (str, int, float)):
        raise ValueError('Observation must be text or a stored number')
    return str(value)


def schema(connection):
    return connection.execute(
        'SELECT type, name, tbl_name, sql FROM sqlite_master '
        'WHERE name NOT LIKE "sqlite_%" ORDER BY type, name').fetchall()


def sample_digest(connection, table='sample'):
    if table not in ('sample', NEW_TABLE):
        raise ValueError('Unexpected table name')
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(
            f'SELECT id_key, observation, point_id, study_id, year '
            f'FROM "{table}" ORDER BY id_key'):
        normalized = (row[0], observation_text(row[1]), *row[2:])
        digest.update((json.dumps(normalized, ensure_ascii=True) + '\n').encode('utf-8'))
        count += 1
    return count, digest.hexdigest()


def check_integrity(connection):
    result = connection.execute('PRAGMA quick_check').fetchall()
    if result != [('ok',)]:
        raise RuntimeError(f'SQLite quick_check failed: {result[:10]}')
    foreign_keys = connection.execute('PRAGMA foreign_key_check').fetchmany(10)
    if foreign_keys:
        raise RuntimeError(f'Foreign key check failed: {foreign_keys}')


def save_report(path, report):
    if path:
        Path(path).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def migrate(database_path, backup_path, report_path=None):
    database_path = Path(database_path).resolve(strict=True)
    backup_path = Path(backup_path).resolve()
    report_path = Path(report_path).resolve() if report_path else None
    if len({database_path, backup_path, report_path} - {None}) != (3 if report_path else 2):
        raise ValueError('Database, backup and report must be different paths')
    if backup_path.exists():
        raise FileExistsError(f'Backup already exists: {backup_path}')
    report = {'database': str(database_path), 'backup': None,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'sqlite_version': sqlite3.sqlite_version, 'status': 'preparing',
              'new_observations_imported': 0}
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    connection = None
    committed = False
    try:
        connection = sqlite3.connect(database_path.as_uri() + '?mode=rw', uri=True, timeout=30)
        connection.execute('PRAGMA foreign_keys=OFF')
        connection.execute('BEGIN IMMEDIATE')
        columns_before = connection.execute('PRAGMA table_info(sample)').fetchall()
        if tuple(row[1] for row in columns_before) != COLUMNS:
            raise ValueError('Unexpected sample schema; migration needs review')
        observation_type = columns_before[1][2].upper()
        if observation_type in ('TEXT', 'VARCHAR'):
            if connection.execute("SELECT count(*) FROM sample WHERE typeof(observation) <> 'text'").fetchone()[0]:
                raise ValueError('Column is text but contains nontext values; review needed')
            report.update(status='already_text', sample_rows=connection.execute('SELECT count(*) FROM sample').fetchone()[0])
            connection.rollback()
            save_report(report_path, report)
            return report
        if observation_type not in ('FLOAT', 'REAL', 'DOUBLE'):
            raise ValueError(f'Unexpected Observation type: {observation_type}')
        before_schema = schema(connection)
        sql = next(row[3] for row in before_schema if row[:2] == ('table', 'sample'))
        if connection.execute('SELECT 1 FROM sqlite_master WHERE name=?', (NEW_TABLE,)).fetchone():
            raise ValueError('Temporary migration table already exists')
        create_sql, n = re.subn(
            r'(?i)\b(observation\s+)(FLOAT|REAL|DOUBLE)\b', r'\1TEXT', sql)
        if n != 1:
            raise ValueError('Could not identify exactly one Observation type declaration')
        create_sql, n = re.subn(
            r'(?i)^CREATE\s+TABLE\s+(?:sample|"sample")(?=\s*\()',
            f'CREATE TABLE "{NEW_TABLE}"', create_sql)
        if n != 1:
            raise ValueError('Unexpected sample CREATE TABLE statement')

        # The reserved writer lock prevents changes between the backup and the
        # migration. A separate reader makes the backup without backing up an
        # active write transaction through the same connection.
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open('xb'):
            pass
        print(f'Backing up {database_path} to {backup_path}', flush=True)
        reader = sqlite3.connect(database_path.as_uri() + '?mode=ro', uri=True)
        destination = sqlite3.connect(backup_path)
        try:
            reader.backup(destination)
            check_integrity(destination)
            report['backup'] = str(backup_path)
        finally:
            destination.close()
            reader.close()
        save_report(report_path, report)

        before_count, before_digest = sample_digest(connection)
        report.update(sample_rows=before_count, normalized_sample_sha256_before=before_digest,
                      original_sample_schema=sql,
                      observation_conversion='Exact strings unchanged; stored numbers converted with Python str for float round-trip precision.')
        print(f'Converting {before_count} existing observations to text', flush=True)
        connection.create_function('observation_as_text', 1, observation_text)
        connection.execute(create_sql)
        connection.execute(
            f'INSERT INTO "{NEW_TABLE}" (id_key, observation, point_id, study_id, year) '
            'SELECT id_key, observation_as_text(observation), point_id, study_id, year FROM sample')
        if sample_digest(connection, NEW_TABLE) != (before_count, before_digest):
            raise RuntimeError('Copied sample values do not match the original')

        # Remove and recreate dependent objects inside the same transaction so
        # the temporary absence of sample never leaves invalid view definitions.
        dependents = [row for row in before_schema if row[0] in ('trigger', 'view')]
        for kind, name, _, _ in dependents:
            connection.execute(f'DROP {kind.upper()} "{name.replace(chr(34), chr(34)*2)}"')
        connection.execute('DROP TABLE sample')
        connection.execute(f'ALTER TABLE "{NEW_TABLE}" RENAME TO sample')
        for kind, _, table, object_sql in before_schema:
            if kind == 'index' and table == 'sample' and object_sql:
                connection.execute(object_sql)
        for _, _, _, object_sql in dependents:
            connection.execute(object_sql)

        columns_after = connection.execute('PRAGMA table_info(sample)').fetchall()
        expected_columns = list(columns_before)
        expected_columns[1] = (*columns_before[1][:2], 'TEXT', *columns_before[1][3:])
        if columns_after != expected_columns:
            raise RuntimeError('Unexpected column or constraint change')
        after_schema = schema(connection)
        if [row for row in after_schema if row[:2] != ('table', 'sample')] != [
                row for row in before_schema if row[:2] != ('table', 'sample')]:
            raise RuntimeError('Unexpected change to another schema object')
        after_count, after_digest = sample_digest(connection)
        if (after_count, after_digest) != (before_count, before_digest):
            raise RuntimeError('Existing sample rows or values changed')
        if connection.execute("SELECT count(*) FROM sample WHERE typeof(observation) <> 'text'").fetchone()[0]:
            raise RuntimeError('A migrated observation is not stored as text')
        check_integrity(connection)
        report.update(normalized_sample_sha256_after=after_digest,
                      final_sample_schema=next(row[3] for row in after_schema if row[:2] == ('table', 'sample')),
                      indexes_triggers_views_preserved=True, quick_check='ok', foreign_key_check='ok')
        connection.commit()
        committed = True
        report.update(status='migrated', completed_utc=datetime.now(timezone.utc).isoformat())
        save_report(report_path, report)
        return report
    except Exception as error:
        if committed:
            # A reporting failure cannot roll back an already committed change.
            report.update(status='migrated_report_error', report_error=str(error))
            try:
                save_report(report_path, report)
            except OSError:
                pass
            return report
        if connection and connection.in_transaction:
            connection.rollback()
        report.update(status='failed', error=str(error))
        save_report(report_path, report)
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
    result = migrate(args.database, args.backup, args.report)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
