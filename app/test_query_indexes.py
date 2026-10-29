"""Check the ORM index schema using isolated, in-memory SQLite databases."""
from pathlib import Path
import runpy
import unittest

from sqlalchemy import create_engine, func, inspect, select

from database_model_definitions import (
    Base, CovariateDefn, CovariateValue, Point, Sample, Study,
)


EXPECTED_INDEXES = {
    'idx_covariate_value_defn_value_sample': (
        'covariate_value', ('covariate_defn_id', 'value', 'sample_id')),
    'idx_sample_study_point': ('sample', ('study_id', 'point_id')),
}


class QueryIndexTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_fresh_orm_schema_has_named_nonunique_indexes_in_column_order(self):
        inspector = inspect(self.engine)
        for name, (table, columns) in EXPECTED_INDEXES.items():
            with self.subTest(index=name):
                indexes = {index['name']: index for index in inspector.get_indexes(table)}
                self.assertIn(name, indexes)
                self.assertEqual(tuple(indexes[name]['column_names']), columns)
                self.assertFalse(indexes[name]['unique'])

    def test_migration_and_fresh_orm_schema_agree(self):
        # This standalone migration only imports the standard library. Loading
        # its definitions must not import app.database or start application services.
        migration = Path(__file__).with_name('migrations') / 'query_indexes.py'
        definitions = runpy.run_path(str(migration))['INDEX_DEFINITIONS']
        self.assertEqual(definitions, EXPECTED_INDEXES)
        for name, (table, columns) in definitions.items():
            with self.subTest(index=name):
                index = next(index for index in Base.metadata.tables[table].indexes
                             if index.name == name)
                self.assertEqual(tuple(index.columns.keys()), columns)

    def test_covariate_filter_and_distinct_site_results_are_unchanged(self):
        with self.engine.begin() as connection:
            connection.execute(Study.__table__.insert(), [
                {'id_key': 1, 'name': 'two-sites'},
                {'id_key': 2, 'name': 'one-site'},
                {'id_key': 3, 'name': 'other-site'},
            ])
            connection.execute(Point.__table__.insert(), [
                {'id_key': 11, 'latitude': 41.0, 'longitude': 2.0},
                {'id_key': 12, 'latitude': 41.1, 'longitude': 2.1},
                {'id_key': 13, 'latitude': 41.2, 'longitude': 2.2},
            ])
            samples = [
                (101, 1, 11, 2020, '<10'),
                (102, 1, 11, 2020, 'NA'),
                (103, 1, 12, 2021, '>50'),
                (104, 2, 11, 2020, 'na'),
                (105, 2, 11, 2021, '1.00'),
                (106, 3, 13, 2022, '>10 and <50'),
            ]
            connection.execute(Sample.__table__.insert(), [
                dict(zip(('id_key', 'study_id', 'point_id', 'year', 'observation'), row))
                for row in samples
            ])
            connection.execute(CovariateDefn.__table__.insert(), [{
                'id_key': 7, 'name': 'test_taxon', 'editable_name': True,
                'queryable': True, 'always_display': False, 'hidden': False,
                'show_in_point_table': False, 'search_by_unique': True,
                'covariate_type': 0, 'covariate_association': 1,
            }])
            # Repeated values and repeated sites must remain allowed. The
            # covariate IN subquery must not duplicate a sample with two matches.
            covariates = [(101, 'aphid'), (101, 'aphid'), (102, 'aphid'),
                          (103, 'beetle'), (104, 'aphid'), (105, 'moth'),
                          (106, 'beetle')]
            connection.execute(CovariateValue.__table__.insert(), [
                {'id_key': number, 'covariate_defn_id': 7,
                 'sample_id': sample_id, 'study_id': None, 'value': value}
                for number, (sample_id, value) in enumerate(covariates, 501)
            ])
            matching_samples = select(CovariateValue.sample_id).where(
                CovariateValue.covariate_defn_id == 7,
                CovariateValue.value.in_(['aphid', 'beetle']))
            matching_studies = select(Sample.study_id).group_by(Sample.study_id).having(
                func.count(func.distinct(Sample.point_id)) >= 2)
            filtered = select(Sample.id_key, Sample.observation).where(
                Sample.id_key.in_(matching_samples)).order_by(Sample.id_key)
            multi_site = filtered.where(Sample.study_id.in_(matching_studies))
            site_counts = select(
                Sample.study_id, func.count(func.distinct(Sample.point_id))
            ).group_by(Sample.study_id).order_by(Sample.study_id)

            def results():
                return {
                    'filtered': [tuple(row) for row in connection.execute(filtered)],
                    'multi_site': [tuple(row) for row in connection.execute(multi_site)],
                    'site_counts': [tuple(row) for row in connection.execute(site_counts)],
                }

            connection.exec_driver_sql('ANALYZE')
            indexed = results()
            self.assertEqual(indexed, {
                'filtered': [(101, '<10'), (102, 'NA'), (103, '>50'),
                             (104, 'na'), (106, '>10 and <50')],
                'multi_site': [(101, '<10'), (102, 'NA'), (103, '>50')],
                'site_counts': [(1, 2), (2, 1), (3, 1)],
            })
            for name, (table, _) in EXPECTED_INDEXES.items():
                index = next(index for index in Base.metadata.tables[table].indexes
                             if index.name == name)
                index.drop(connection)
            connection.exec_driver_sql('ANALYZE')
            self.assertEqual(results(), indexed)


if __name__ == '__main__':
    unittest.main()
