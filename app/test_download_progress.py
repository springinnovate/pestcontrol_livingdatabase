"""Check download progress without starting Celery, Redis or the Flask app."""
import ast
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call
import zipfile

from database_model_definitions import Point, Sample, Study
import test_observation_text as observation_fixture


class DownloadProgressTest(unittest.TestCase):
    def setUp(self):
        # Reuse its isolated database/function setup without inheriting and
        # rediscovering the observation test methods in this module.
        self.fixture = observation_fixture.ObservationTextTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.scope = self.fixture.scope
        self.task_id = 'download-test-task'
        self.task_metadata = {'status': 'PENDING', 'result': None}
        self.task = SimpleNamespace(id=self.task_id, backend=SimpleNamespace(
            get_task_meta=lambda task_id: self.task_metadata))
        self.worker = self.scope['_prep_download']
        self.worker.AsyncResult = lambda task_id: self.task
        self.scope['url_for'] = lambda endpoint, task_id: f'/{endpoint}/{task_id}'
        module = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        endpoint = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'check_task')
        endpoint.decorator_list = []
        exec(compile(ast.Module(body=[endpoint], type_ignores=[]),
                     'app.py isolated check_task', 'exec'), self.scope)

    def check_task(self, state, info):
        self.task_metadata.update(status=state, result=info)
        # Assert against the endpoint's JSON payload, not the worker's metadata.
        result = json.loads(json.dumps(self.scope['check_task'](self.task_id)))
        self.assertEqual(result['status'], state)
        self.assertEqual(result['result_url'], '/download_pcld_points/' + self.task_id)
        return result

    def test_each_poll_reads_one_consistent_status_and_result_snapshot(self):
        read = Mock(side_effect=[
            {'status': 'PROGRESS', 'result': {'current': 1000, 'total': 1003}},
            {'status': 'SUCCESS', 'result': {'zipfile_path': 'result.zip'}},
        ])
        self.task.backend.get_task_meta = read
        progress = self.scope['check_task'](self.task_id)
        self.assertEqual(progress['status'], 'PROGRESS')
        self.assertEqual((progress['current'], progress['total']), (1000, 1003))
        read.assert_called_once_with(self.task_id)

        success = self.scope['check_task'](self.task_id)
        self.assertEqual(success['status'], 'SUCCESS')
        self.assertNotIn('current', success)
        self.assertEqual(read.call_args_list, [call(self.task_id), call(self.task_id)])

    def test_progress_preserves_zero_numeric_totals_and_unknown_total_marker(self):
        for current, total in [(0, 1003), (1000, 1003), (0, '-'), (1000, '-')]:
            with self.subTest(current=current, total=total):
                result = self.check_task('PROGRESS', {'current': current, 'total': total})
                self.assertEqual(result['current'], current)
                self.assertEqual(result['total'], total)

    def test_missing_or_malformed_progress_metadata_raises(self):
        for info in [None, {}, 'not progress metadata', [], RuntimeError('unavailable'),
                     {'current': 0}, {'total': 1003}]:
            with self.subTest(info=info):
                with self.assertRaises((KeyError, TypeError)):
                    self.check_task('PROGRESS', info)

    def test_invalid_progress_counts_raise_instead_of_returning_defaults(self):
        for field in ['current', 'total']:
            for invalid in [None, '1003', True, -1, 1.5, float('nan'), float('inf')]:
                with self.subTest(field=field, invalid=invalid):
                    info = {'current': 0, 'total': 1003, field: invalid}
                    with self.assertRaises(ValueError):
                        self.check_task('PROGRESS', info)
        with self.assertRaises(ValueError):
            self.check_task('PROGRESS', {'current': '-', 'total': 1003})

    def test_nonprogress_states_keep_status_and_download_url_without_inventing_counts(self):
        for state, info in [
                ('PENDING', None),
                ('STARTED', {'pid': 123}),
                ('SUCCESS', {'query_id': 'test-query', 'zipfile_path': 'result.zip'})]:
            with self.subTest(state=state):
                result = self.check_task(state, info)
                self.assertIsNone(result.get('current'))
                self.assertIsNone(result.get('total'))
        result = self.check_task('FAILURE', RuntimeError('download failed'))
        self.assertEqual(result['error'], 'download failed')

    def test_real_worker_progress_reaches_status_endpoint_and_zip_rows_stay_exact(self):
        expected = (self.fixture.values * 126)[:1003]
        with self.fixture.sessions() as session:
            point = session.query(Point).one()
            study = session.query(Study).one()
            session.add_all(Sample(observation=value, point=point, study=study, year=2026)
                            for value in expected[len(self.fixture.values):])
            session.commit()

        # An expired count cache reports '-' while the same export still runs.
        for cached_count, expected_total in [(1003, 1003), (None, '-')]:
            with self.subTest(cached_count=cached_count), tempfile.TemporaryDirectory() as directory:
                self.scope['QUERY_RESULTS_DIR'] = directory
                self.scope['redis_client'].set(
                    'sample_count:test-query',
                    json.dumps({'sample_count': cached_count}) if cached_count is not None else None)
                progress = []

                def update_state(state, meta):
                    response = self.check_task(state, meta)
                    progress.append((response['current'], response['total']))

                result = self.worker(SimpleNamespace(update_state=update_state), 'test-query')
                self.assertEqual(progress, [(0, expected_total), (1000, expected_total)])
                with zipfile.ZipFile(result['zipfile_path']) as archive:
                    sample_csv = archive.read('sample_data_test-query.csv').decode('utf-8')
                    self.assertIn('study_data_test-query.csv', archive.namelist())
                actual = [row['observation'] for row in csv.DictReader(io.StringIO(sample_csv))]
                self.assertEqual(actual, expected)
                success = self.check_task('SUCCESS', result)
                self.assertIsNone(success.get('current'))
                self.assertIsNone(success.get('total'))


if __name__ == '__main__':
    unittest.main()
