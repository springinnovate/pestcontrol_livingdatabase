"""Keep supplied observation text intact through storage and query exports.

Load only the endpoint functions, so tests never start Redis, Celery, GEE,
or the configured database connection.
"""
import ast
import csv
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import pandas as pd
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import selectinload, sessionmaker

from database_model_definitions import (
    Base, CovariateDefn, CovariateValue, Point, Sample, Study,
)


class ObservationTextTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.values = ['<10', 'NA', '>50', '10-50', '0', '1.00', '001', '>=2']
        with self.sessions() as session:
            point = Point(latitude=41.5, longitude=2.1)
            study = Study(name='text-observations')
            session.add_all([
                Sample(observation=value, point=point, study=study, year=2026)
                for value in self.values
            ])
            session.commit()
        cache = {'test-query': '{}'}
        self.scope = {
            'SessionLocal': self.sessions,
            'Sample': Sample, 'Study': Study, 'Point': Point,
            'CovariateDefn': CovariateDefn, 'CovariateValue': CovariateValue,
            'selectinload': selectinload, 'func': func,
            'build_filter': lambda session, form: ([], ''),
            'redis_client': SimpleNamespace(get=cache.get, set=cache.__setitem__),
            'request': SimpleNamespace(args={'query_id': 'test-query'}),
            'jsonify': lambda value: json.loads(json.dumps(value)),
            'LOGGER': logging.getLogger(__name__),
            'json': json, 'os': os, 'pd': pd, 'zipfile': zipfile,
            'delete_file': SimpleNamespace(apply_async=lambda **kwargs: None),
        }
        module = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        functions = []
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in {'get_data', '_prep_download'}:
                node.decorator_list = []
                functions.append(node)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'app.py', 'exec'), self.scope)

    def tearDown(self):
        self.engine.dispose()

    def test_storage_and_query_json_preserve_exact_text(self):
        with self.engine.connect() as connection:
            column_types = {row[1]: row[2] for row in connection.execute(text('PRAGMA table_info(sample)'))}
            self.assertEqual(column_types['observation'], 'TEXT')
            storage_types = list(connection.execute(text('SELECT DISTINCT typeof(observation) FROM sample')).scalars())
            self.assertEqual(storage_types, ['text'])
        response = self.scope['get_data']()
        self.assertEqual([row['observation'] for row in response['sample_data_rows']], self.values)

    def test_download_csv_preserves_exact_text_in_full_and_partial_batches(self):
        # Exercise the normal 1,000-row flush as well as the final partial batch.
        expected = (self.values * 126)[:1003]
        with self.sessions() as session:
            point = session.query(Point).one()
            study = session.query(Study).one()
            session.add_all([
                Sample(observation=value, point=point, study=study, year=2026)
                for value in expected[len(self.values):]
            ])
            session.commit()
        with tempfile.TemporaryDirectory() as directory:
            self.scope['QUERY_RESULTS_DIR'] = directory
            result = self.scope['_prep_download'](
                SimpleNamespace(update_state=lambda **kwargs: None), 'test-query')
            with zipfile.ZipFile(result['zipfile_path']) as archive:
                data = archive.read('sample_data_test-query.csv').decode('utf-8')
            actual = [row['observation'] for row in csv.DictReader(io.StringIO(data))]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
