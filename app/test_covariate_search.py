"""Covariate options and submitted filters work without starting app services."""
import ast
import collections
import json
import logging
import os
from pathlib import Path
import pickle
import re
import tempfile
import unittest
from urllib.parse import parse_qs, urlencode

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import and_, create_engine, distinct, func, select
from sqlalchemy.orm import sessionmaker

from database_model_definitions import (
    Base, CovariateAssociation, CovariateDefn, CovariateType, CovariateValue,
    Geolocation, Point, Sample, Study, STUDY_ID,
)


APP_DIR = Path(__file__).resolve().parent
SPECIAL_FIELD = 'Covariate_"plot\\name<&>'
SPECIAL_VALUE = 'O\'Brien "north" \\path\nnext\rline & <edge> &amp; </script>\u2028\u2029'


class EncodedForm:
    """The get/getlist interface used by form_to_dict, after form decoding."""

    def __init__(self, values):
        self.values = parse_qs(urlencode(values, doseq=True), keep_blank_values=True)

    def get(self, name):
        values = self.values.get(name, [])
        return values[0] if values else None

    def getlist(self, name):
        return self.values.get(name, [])


class CovariateSearchTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.sessions = []
        self.temp = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.temp.name) / 'initialize_searchable_covariates.pkl'
        self.scope = {
            'SessionLocal': self.session,
            'INSTANCE_DIR': self.temp.name,
            'COVARIATE_CACHE_VERSION': 1,
            'COVARIATE_STATE': {},
            'CovariateDefn': CovariateDefn, 'CovariateValue': CovariateValue,
            'CovariateType': CovariateType, 'CovariateAssociation': CovariateAssociation,
            'Sample': Sample, 'Study': Study, 'Point': Point, 'Geolocation': Geolocation,
            'STUDY_ID': STUDY_ID, 'collections': collections, 'os': os,
            'pickle': pickle, 'func': func, 'distinct': distinct,
            'select': select, 'and_': and_, 're': re,
            'LOGGER': logging.getLogger(__name__),
        }
        source = ast.parse((APP_DIR / 'app.py').read_text(encoding='utf-8'))
        names = {'initialize_covariates', 'form_to_dict', 'build_filter', 'extract_years'}
        functions = [node for node in source.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in functions}, names)
        for node in functions:
            node.decorator_list = []
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'app.py isolated functions', 'exec'), self.scope)
        self.add_fixture()

    def session(self):
        session = self.session_factory()
        self.sessions.append(session)
        return session

    def tearDown(self):
        for session in self.sessions:
            session.close()
        self.engine.dispose()
        self.temp.cleanup()

    def add_fixture(self):
        with self.session_factory() as session:
            points = [Point(id_key=1, latitude=41, longitude=2),
                      Point(id_key=2, latitude=42, longitude=3)]
            studies = [Study(id_key=1, name='Zulu'), Study(id_key=2, name='Alpha')]
            session.add_all(points + studies)
            session.add_all([
                Sample(id_key=number, observation=str(number), point_id=point,
                       study_id=study, year=2020)
                for number, study, point in [(11, 1, 1), (12, 1, 2), (13, 2, 1), (14, 2, 2)]
            ])
            definitions = [
                (1, 'Taxon', CovariateType.STRING, False, True, 1),
                (2, 'Temperature', CovariateType.FLOAT, False, True, 1),
                (3, 'Replicate', CovariateType.INTEGER, False, True, 1),
                (4, 'Units', CovariateType.STRING, True, True, 1),
                (5, SPECIAL_FIELD, CovariateType.STRING, True, True, 1),
                (6, 'Not_queryable', CovariateType.STRING, True, False, 1),
                (7, 'Empty_queryable', CovariateType.STRING, False, True, 1),
                (8, 'Sampling_Design', CovariateType.STRING, False, True, 0),
                (9, 'Literal_type_name', CovariateType.STRING, False, True, 1),
            ]
            for number, name, data_type, unique, queryable, association in definitions:
                session.add(CovariateDefn(
                    id_key=number, name=name, editable_name=True, display_order=number,
                    queryable=queryable, always_display=False, hidden=False,
                    show_in_point_table=False, search_by_unique=unique,
                    covariate_type=data_type.value, covariate_association=association))
            sample_values = {
                1: [(11, 'aphid'), (11, 'aphid'), (12, 'beetle'), (13, 'aphid'), (14, 'beetle')],
                2: [(11, '10.50'), (12, '2.0'), (13, '10.50')],
                3: [(11, '001'), (12, '2'), (13, '001')],
                4: [(11, 'count'), (12, 'count'), (13, 'per plant'), (14, 'count')],
                5: [(11, SPECIAL_VALUE)],
                6: [(11, 'must not be offered')],
                9: [(11, 'STRING')],
            }
            for definition, values in sample_values.items():
                session.add_all(CovariateValue(covariate_defn_id=definition,
                    sample_id=sample_id, value=value) for sample_id, value in values)
            session.add_all([
                CovariateValue(covariate_defn_id=8, study_id=1, value='Organic'),
                CovariateValue(covariate_defn_id=8, study_id=2, value='Conventional'),
            ])
            session.commit()

    def initialize(self, clear_cache=False):
        self.scope['initialize_covariates'](clear_cache)
        return self.scope['COVARIATE_STATE']

    def assert_actual_options(self, state):
        values = state['searchable_covariates']
        self.assertEqual(values['Taxon'], ['aphid', 'beetle'])
        self.assertEqual(values['Temperature'], ['10.50', '2.0'])
        self.assertEqual(values['Replicate'], ['001', '2'])
        self.assertEqual(values['Units'], ['count', 'per plant'])
        self.assertEqual(values[SPECIAL_FIELD], [SPECIAL_VALUE])
        self.assertEqual(values['Empty_queryable'], [])
        self.assertEqual(values['Sampling_Design'], ['Conventional', 'Organic'])
        self.assertEqual(values['Literal_type_name'], ['STRING'])
        self.assertEqual(values[STUDY_ID], ['Alpha', 'Zulu'])
        self.assertNotIn('Not_queryable', values)
        self.assertTrue(all(isinstance(options, list) for options in values.values()))
        self.assertEqual(state['cache_version'], 1)

    def test_cold_cache_uses_sorted_distinct_stored_values_for_every_type(self):
        state = self.initialize()
        self.assert_actual_options(state)
        with self.cache_path.open('rb') as stream:
            cached = pickle.load(stream)
        # The on-disk state must already match the rendered state. Previously
        # type markers were pickled before they became sorted character lists.
        self.assertEqual(cached, state)
        self.assert_actual_options(cached)

    def test_warm_cache_returns_the_same_options_without_database_access(self):
        expected = self.initialize()
        self.scope['COVARIATE_STATE'] = {}

        def no_database_access():
            self.fail('A valid warm cache should not query the database')

        self.scope['SessionLocal'] = no_database_access
        actual = self.initialize()
        self.assertEqual(actual, expected)
        self.assert_actual_options(actual)

    def test_explicit_cache_refresh_includes_new_stored_values(self):
        self.assert_actual_options(self.initialize())
        with self.session_factory() as session:
            session.add(CovariateValue(covariate_defn_id=1, sample_id=14, value='lacewing'))
            session.commit()
        refreshed = self.initialize(clear_cache=True)
        with self.cache_path.open('rb') as stream:
            cached = pickle.load(stream)
        self.assertEqual(refreshed, cached)
        for state in [refreshed, cached]:
            values = state['searchable_covariates']
            self.assertEqual(values['Taxon'], ['aphid', 'beetle', 'lacewing'])
            self.assertEqual(values['Temperature'], ['10.50', '2.0'])
            self.assertEqual(values['Replicate'], ['001', '2'])
            self.assertEqual(values['Literal_type_name'], ['STRING'])
            self.assertTrue(all(isinstance(options, list) for options in values.values()))

    def test_legacy_type_marker_and_character_caches_are_rebuilt(self):
        for legacy_value in ['STRING', ['G', 'I', 'N', 'R', 'S', 'T']]:
            with self.subTest(legacy_value=legacy_value):
                legacy = {'searchable_covariates': {'Taxon': legacy_value},
                          'country_set': [], 'continent_set': [],
                          'covariate_defns': [], 'n_samples': 0}
                with self.cache_path.open('wb') as stream:
                    pickle.dump(legacy, stream)
                state = self.initialize()
                self.assert_actual_options(state)
                with self.cache_path.open('rb') as stream:
                    self.assert_actual_options(pickle.load(stream))

    def test_template_serializes_values_as_safe_round_trippable_json(self):
        state = self.initialize()
        environment = Environment(
            loader=FileSystemLoader(APP_DIR / 'templates'),
            autoescape=select_autoescape(['html']))
        environment.globals['url_for'] = lambda endpoint, **kwargs: '/' + endpoint
        rendered = environment.get_template('query_builder.html').render(
            unique_covariate_values=state['searchable_covariates'],
            possible_operations=['='], country_set=[], continent_set=[])
        match = re.search(r'const fieldValues\s*=\s*(.*?);\s*function updateValueOptions',
                          rendered, flags=re.DOTALL)
        self.assertIsNotNone(match)
        payload = match.group(1)
        self.assertNotIn('</script>', payload.lower())
        self.assertEqual(json.loads(payload), state['searchable_covariates'])
        self.assertEqual(json.loads(payload)[SPECIAL_FIELD], [SPECIAL_VALUE])

    def test_selected_values_survive_form_submission_and_filter_actual_rows(self):
        options = self.initialize()['searchable_covariates']
        cases = [
            ([('Taxon', ['aphid'])], [11, 13]),
            ([('Taxon', ['aphid', 'beetle']), ('Units', ['count'])], [11, 12, 14]),
            ([('Sampling_Design', ['Organic']), ('Taxon', ['beetle'])], [12]),
            ([(STUDY_ID, ['Alpha'])], [13, 14]),
            ([(SPECIAL_FIELD, [SPECIAL_VALUE])], [11]),
        ]
        for selections, expected in cases:
            with self.subTest(selections=selections):
                for field, selected in selections:
                    self.assertTrue(set(selected).issubset(options[field]))
                fields = {
                    'covariate': [field for field, _ in selections],
                    'valueCounts': [len(values) for _, values in selections],
                    'values': [value for _, values in selections for value in values],
                    'minSitesPerStudy': '0', 'sampleSizeMinYears': '0',
                    'sampleSizeMinObservationsPerYear': '0',
                }
                parsed = self.scope['form_to_dict'](EncodedForm(fields))
                self.assertEqual(parsed['values'], [values for _, values in selections])
                with self.session_factory() as session:
                    filters, _ = self.scope['build_filter'](session, parsed)
                    actual = [number for number, in session.query(Sample.id_key)
                              .join(Sample.study).join(Sample.point)
                              .filter(*filters).order_by(Sample.id_key)]
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
