"""Exercise the manual index/statistics migration on isolated SQLite fixtures."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import query_indexes


class QueryIndexMigrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / 'database.db'
        self.backup = self.root / 'backup.db'
        self.report = self.root / 'report.json'
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.executescript('''
                CREATE TABLE sample(id_key INTEGER PRIMARY KEY,observation TEXT,study_id INTEGER,point_id INTEGER,year INTEGER);
                CREATE TABLE covariate_value(id_key INT,value TEXT,covariate_defn_id INT,study_id INT,sample_id INT);
                CREATE TABLE other(value TEXT);
                CREATE INDEX old_sample_study ON sample(study_id);
                CREATE INDEX old_sample_study_year ON sample(study_id,year);
                CREATE INDEX old_covariate_defn_value ON covariate_value(covariate_defn_id,value);
                CREATE VIEW observations AS SELECT id_key,observation FROM sample;
                CREATE TRIGGER sample_audit AFTER INSERT ON sample BEGIN INSERT INTO other VALUES('inserted'); END;
                INSERT INTO sample VALUES(1,'<10',1,10,2020),(2,'NA',1,11,2021);
                INSERT INTO covariate_value VALUES(NULL,'literal',3,NULL,1),(11,'  ',3,NULL,2);
                ANALYZE;
                INSERT INTO sample VALUES(3,'001',2,20,2022),(4,'>10 and <50',2,21,2022),(5,'#REF!',2,20,2023);
                INSERT INTO covariate_value VALUES(12,'abundance',3,NULL,3),(13,'NA',3,NULL,4),(14,'study metadata',4,2,NULL);
            ''')

    def snapshot(self, path):
        with closing(sqlite3.connect(path)) as connection:
            return {
                'schema': query_indexes.schema(connection),
                'statistics': query_indexes.statistics(connection),
                'rows': {table: connection.execute(f'SELECT rowid,* FROM {table} ORDER BY rowid').fetchall()
                         for table in ('sample', 'covariate_value', 'other')},
            }

    def run_migration(self, backup=None, report=None):
        return query_indexes.migrate(self.database, backup or self.backup, report or self.report)

    def test_creates_exact_indexes_preserves_rows_backup_and_refreshes_all_statistics(self):
        before = self.snapshot(self.database)
        result = self.run_migration()
        after = self.snapshot(self.database)
        self.assertEqual(result['status'], 'migrated')
        self.assertEqual(result['analyze'], 'full')
        self.assertEqual(set(result['indexes_created']), set(query_indexes.INDEX_DEFINITIONS))
        self.assertEqual(result['indexes_reused'], [])
        self.assertEqual(after['rows'], before['rows'])
        self.assertEqual(self.snapshot(self.backup), before)
        old_statistics = {row[1]: row[2] for row in before['statistics']}
        new_statistics = {row[1]: row[2] for row in after['statistics']}
        for name in ('old_sample_study', 'old_sample_study_year', 'old_covariate_defn_value'):
            self.assertEqual(old_statistics[name].split()[0], '2')
            self.assertEqual(new_statistics[name].split()[0], '5')
        for name in query_indexes.INDEX_DEFINITIONS:
            self.assertEqual(new_statistics[name].split()[0], '5')
        self.assertEqual([row for row in after['schema'] if row[1] not in query_indexes.INDEX_DEFINITIONS], before['schema'])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(set(query_indexes.validate_indexes(connection)), set(query_indexes.INDEX_DEFINITIONS))
            connection.execute("INSERT INTO sample VALUES(6,'unchanged trigger',2,21,2024)")
            self.assertEqual(connection.execute('SELECT count(*) FROM other').fetchone()[0], 6)
        self.assertEqual(json.loads(self.report.read_text())['status'], 'migrated')

    def test_repeat_creates_new_backup_and_refreshes_existing_and_target_statistics(self):
        self.run_migration()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO sample VALUES(6,'0',3,30,2024)")
            connection.execute("INSERT INTO covariate_value VALUES(15,'zero',3,NULL,6)")
        before = self.snapshot(self.database)
        backup = self.root / 'second-backup.db'
        report = self.root / 'second-report.json'
        result = self.run_migration(backup, report)
        after = self.snapshot(self.database)
        self.assertEqual(result['status'], 'migrated')
        self.assertEqual(result['indexes_created'], [])
        self.assertEqual(set(result['indexes_reused']), set(query_indexes.INDEX_DEFINITIONS))
        self.assertEqual(after['rows'], before['rows'])
        self.assertEqual(after['schema'], before['schema'])
        self.assertEqual(self.snapshot(backup), before)
        for table, _, stat in after['statistics']:
            if table in ('sample', 'covariate_value'):
                self.assertEqual(stat.split()[0], '6')

    def test_rejects_conflicting_index_definitions_before_backup(self):
        name = 'idx_covariate_value_defn_value_sample'
        statements = [
            f'CREATE INDEX {name} ON covariate_value(value,covariate_defn_id,sample_id)',
            f'CREATE UNIQUE INDEX {name} ON covariate_value(covariate_defn_id,value,sample_id)',
            f'CREATE INDEX {name} ON covariate_value(covariate_defn_id,value,sample_id) WHERE sample_id IS NOT NULL',
            f'CREATE INDEX {name} ON covariate_value(covariate_defn_id,value DESC,sample_id)',
            f'CREATE INDEX {name} ON covariate_value(covariate_defn_id,value COLLATE NOCASE,sample_id)',
            f'CREATE INDEX {name} ON sample(study_id,point_id)',
        ]
        for number, statement in enumerate(statements):
            with self.subTest(statement=statement):
                with closing(sqlite3.connect(self.database)) as connection, connection:
                    connection.execute(f'DROP INDEX IF EXISTS {name}')
                    connection.execute(statement)
                before = self.snapshot(self.database)
                with self.assertRaisesRegex(ValueError, 'Conflicting'):
                    self.run_migration(self.root / f'backup-{number}.db', self.root / f'report-{number}.json')
                self.assertEqual(self.snapshot(self.database), before)
                self.assertFalse((self.root / f'backup-{number}.db').exists())

    def test_failure_after_analyze_rolls_back_indexes_statistics_and_preserves_backup(self):
        before = self.snapshot(self.database)
        original_check = query_indexes.check_integrity
        calls = 0
        def fail_after_analyze(connection):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertEqual(len(query_indexes.validate_indexes(connection)), 2)
                self.assertNotEqual(query_indexes.statistics(connection), before['statistics'])
                raise RuntimeError('Simulated validation failure after ANALYZE')
            return original_check(connection)
        with mock.patch.object(query_indexes, 'check_integrity', fail_after_analyze):
            with self.assertRaisesRegex(RuntimeError, 'after ANALYZE'):
                self.run_migration()
        self.assertEqual(self.snapshot(self.database), before)
        self.assertEqual(self.snapshot(self.backup), before)
        self.assertEqual(json.loads(self.report.read_text())['status'], 'failed')

    def test_rejects_existing_backup_report_and_reused_paths(self):
        before = self.snapshot(self.database)
        self.backup.write_bytes(b'keep backup')
        with self.assertRaises(FileExistsError):
            self.run_migration()
        self.assertEqual(self.backup.read_bytes(), b'keep backup')
        self.report.write_bytes(b'keep report')
        with self.assertRaises(FileExistsError):
            self.run_migration(self.root / 'fresh.db')
        self.assertEqual(self.report.read_bytes(), b'keep report')
        for backup, report in [(self.database, self.root / 'fresh.json'),
                               (self.root / 'fresh.db', self.database),
                               (self.root / 'same', self.root / 'same')]:
            with self.assertRaises(ValueError):
                self.run_migration(backup, report)
        self.assertEqual(self.snapshot(self.database), before)

    def test_post_commit_report_failure_reports_migrated_state(self):
        before = self.snapshot(self.database)
        original_save = query_indexes.save_report
        def fail_final_report(path, report):
            if report['status'] == 'migrated':
                raise OSError('Simulated final report failure')
            return original_save(path, report)
        with mock.patch.object(query_indexes, 'save_report', fail_final_report):
            result = self.run_migration()
        self.assertEqual(result['status'], 'migrated_report_error')
        after = self.snapshot(self.database)
        self.assertEqual(after['rows'], before['rows'])
        self.assertNotEqual(after['statistics'], before['statistics'])
        self.assertEqual(self.snapshot(self.backup), before)
        self.assertEqual(json.loads(self.report.read_text())['status'], 'migrated_report_error')


if __name__ == '__main__':
    unittest.main()
