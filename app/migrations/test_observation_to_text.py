import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import observation_to_text

from observation_to_text import migrate


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'data.db'
        self.backup = self.root / 'before.db'
        self.report = self.root / 'report.json'
        with closing(sqlite3.connect(self.db)) as c, c:
            c.executescript('''
                CREATE TABLE point(id_key INTEGER PRIMARY KEY);
                CREATE TABLE study(id_key INTEGER PRIMARY KEY);
                CREATE TABLE sample(
                    id_key INTEGER NOT NULL PRIMARY KEY,
                    observation FLOAT NOT NULL,
                    point_id INTEGER NOT NULL REFERENCES point(id_key),
                    study_id INTEGER NOT NULL REFERENCES study(id_key),
                    year INTEGER NOT NULL);
                CREATE INDEX sample_year ON sample(year);
                CREATE INDEX sample_observation ON sample(observation);
                CREATE TABLE related(sample_id INTEGER REFERENCES sample(id_key) ON DELETE CASCADE);
                CREATE TABLE audit(message TEXT);
                CREATE TRIGGER sample_insert AFTER INSERT ON sample
                    BEGIN INSERT INTO audit VALUES('inserted'); END;
                CREATE VIEW sample_view AS SELECT id_key, observation FROM sample;
                INSERT INTO point VALUES(1);
                INSERT INTO study VALUES(1);
            ''')
            c.executemany('INSERT INTO sample VALUES(?, ?, 1, 1, 2020)',
                          enumerate([0.0, 0.12345678901234566, 1.2345678901234567e100, '<10', 'NA', '>10 and <50', '>50'], 1))
            c.execute('INSERT INTO related VALUES(1)')

    def test_migration_preserves_values_dependencies_and_backup(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            before = c.execute('SELECT * FROM sample ORDER BY id_key').fetchall()
        result = migrate(self.db, self.backup, self.report)
        self.assertEqual(result['status'], 'migrated')
        with closing(sqlite3.connect(self.db)) as c, c:
            after = c.execute('SELECT * FROM sample ORDER BY id_key').fetchall()
            self.assertEqual(after, [(r[0], str(r[1]), *r[2:]) for r in before])
            self.assertEqual(c.execute('SELECT typeof(observation), count(*) FROM sample GROUP BY 1').fetchall(), [('text', 7)])
            self.assertEqual(c.execute('SELECT * FROM related').fetchall(), [(1,)])
            self.assertEqual(c.execute('SELECT count(*) FROM audit').fetchone()[0], 7)
            self.assertEqual(c.execute('SELECT count(*) FROM sample_view').fetchone()[0], 7)
            c.execute("INSERT INTO sample VALUES(8, ' NA ', 1, 1, 2020)")
            self.assertEqual(c.execute('SELECT count(*) FROM audit').fetchone()[0], 8)
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
        with closing(sqlite3.connect(self.backup)) as c, c:
            self.assertEqual(c.execute('SELECT * FROM sample ORDER BY id_key').fetchall(), before)
            self.assertEqual(c.execute('PRAGMA table_info(sample)').fetchall()[1][2], 'FLOAT')

    def test_repeat_is_noop_and_does_not_duplicate_rows(self):
        migrate(self.db, self.backup, self.report)
        result = migrate(self.db, self.root/'unused.db', self.root/'second.json')
        self.assertEqual(result['status'], 'already_text')
        self.assertFalse((self.root/'unused.db').exists())

    def test_failed_conversion_rolls_back_and_retains_backup(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('UPDATE sample SET observation=? WHERE id_key=1', (b'unknown',))
        with self.assertRaises(ValueError):
            migrate(self.db, self.backup, self.report)
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('PRAGMA table_info(sample)').fetchall()[1][2], 'FLOAT')
            self.assertEqual(c.execute('SELECT observation FROM sample WHERE id_key=1').fetchone()[0], b'unknown')
        self.assertTrue(self.backup.exists())
        self.assertEqual(json.loads(self.report.read_text())['status'], 'failed')

    def test_existing_backup_is_not_overwritten(self):
        self.backup.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            migrate(self.db, self.backup, self.report)
        self.assertEqual(self.backup.read_bytes(), b'keep')

    def test_failed_validation_after_rebuild_rolls_back_schema_and_rows(self):
        original_check = observation_to_text.check_integrity
        calls = 0
        def fail_second_check(connection):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Simulated validation failure after rebuild')
            return original_check(connection)
        with mock.patch.object(observation_to_text, 'check_integrity', fail_second_check):
            with self.assertRaises(RuntimeError):
                migrate(self.db, self.backup, self.report)
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('PRAGMA table_info(sample)').fetchall()[1][2], 'FLOAT')
            self.assertEqual(c.execute('SELECT count(*) FROM sample_view').fetchone()[0], 7)
            self.assertEqual(c.execute('SELECT * FROM related').fetchall(), [(1,)])

    def test_post_commit_report_failure_does_not_claim_rollback(self):
        original_save = observation_to_text.save_report
        def fail_final_report(path, report):
            if report['status'] == 'migrated':
                raise OSError('Simulated report write failure')
            original_save(path, report)
        with mock.patch.object(observation_to_text, 'save_report', fail_final_report):
            result = migrate(self.db, self.backup, self.report)
        self.assertEqual(result['status'], 'migrated_report_error')
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('PRAGMA table_info(sample)').fetchall()[1][2], 'TEXT')
            self.assertEqual(c.execute('SELECT count(*) FROM sample').fetchone()[0], 7)


if __name__ == '__main__':
    unittest.main()
