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

    def test_download_headers_follow_the_selected_points_and_studies(self):
        with self.sessions() as session:
            point = session.query(Point).one()
            sample = session.query(Sample).first()
            outside = Sample(observation='7', year=2026, study=Study(name='outside'),
                             point=Point(latitude=30, longitude=2))
            no_metadata = Sample(observation='2', year=2026, point=point,
                                 study=Study(name='no-metadata'))
            session.add_all([outside, no_metadata])
            for name, owner, value in [
                ('Sampling_Design', sample.study, 'Transect'),
                ('Paper_DOI', outside.study, 'outside only'),
                ('Response_variable', sample, 'abundance'),
                ('Units', outside, 'per trap'),
            ]:
                field = CovariateDefn(
                    name=name, covariate_association=0 if isinstance(owner, Study) else 1,
                    covariate_type=0, editable_name=False, queryable=False,
                    always_display=False, hidden=False, show_in_point_table=False,
                    search_by_unique=False)
                owner.covariates.append(CovariateValue(covariate_defn=field, value=value))
            session.commit()

        self.scope['build_filter'] = lambda session, form: ([Point.latitude > 40], '')
        with tempfile.TemporaryDirectory() as directory:
            self.scope['QUERY_RESULTS_DIR'] = directory
            result = self.scope['_prep_download'](
                SimpleNamespace(update_state=lambda **kwargs: None), 'test-query')
            with zipfile.ZipFile(result['zipfile_path']) as archive:
                with archive.open('study_data_test-query.csv') as study_csv:
                    studies = csv.DictReader(io.TextIOWrapper(study_csv, encoding='utf-8'))
                    self.assertEqual(studies.fieldnames, ['study_id', 'Sampling_Design'])
                    self.assertEqual({r['study_id']: r['Sampling_Design'] for r in studies},
                                     {'text-observations': 'Transect', 'no-metadata': ''})
                with archive.open('sample_data_test-query.csv') as sample_csv:
                    samples = csv.DictReader(io.TextIOWrapper(sample_csv, encoding='utf-8'))
                    self.assertIn('Response_variable', samples.fieldnames)
                    self.assertNotIn('Units', samples.fieldnames)


if __name__ == '__main__':
    unittest.main()
