import random
import unittest
import numpy as np
import pandas as pd
from utils_test import prod
from unittest.mock import Mock
from sklearn.model_selection import RepeatedStratifiedKFold
from survwrap.optimization import _guess_tries
from survwrap import generate_topology_grid, get_top_models, list_available_datasets, get_data, load_test_data, \
    survival_train_test_split, survival_crossval_splitter, get_time, get_indicator,  dataset, SurvivalEstimator, DeepSurvivalMachines, DeepHitSingle


# OPTIMITAZION.PY
class TestGenerateTopologyGrid(unittest.TestCase):
    def test_output_type(self):
        topologies = generate_topology_grid(5)

        self.assertIsInstance(topologies, list)

    def test_max_layers_default_value(self):
        width= 5
        topologies = generate_topology_grid(width)

        self.assertEqual(len(topologies[-1]), 3)


    def test_custom_max_layers(self):
        width = 5
        max_layers = 2
        topologies = generate_topology_grid(width, max_layers=max_layers)

        self.assertEqual(len(topologies[-1]), max_layers)


    def test_topology_generation(self):
        topologies = generate_topology_grid(5)
        expected_topologies = [[3], [4], [3, 3], [4, 4], [3, 3, 3], [4, 4, 4]]
        self.assertEqual(topologies, expected_topologies)

        topologies = generate_topology_grid(7, max_layers=2)
        expected_topologies = [[3], [4], [5], [7], [3, 3], [4, 4], [5, 5], [7, 7]]
        self.assertEqual(topologies, expected_topologies)


class TestGuessTries(unittest.TestCase):
    def test_empty_grid(self):
        grid = {}
        result = _guess_tries(grid)
        self.assertEqual(result, 1)

    def test_non_empty_grid(self):
        grid = {'a': [1, 2, 3], 'b': [4, 5], 'c': [6]}
        result = _guess_tries(grid)
        expected_result = 1 + int(0.05 * prod([len(_) for _ in grid.values()]))
        self.assertEqual(result, expected_result)

    def test_custom_fraction(self):
        grid = {'a': [1, 2, 3], 'b': [4, 5], 'c': [6]}
        fraction = 0.1
        result = _guess_tries(grid, fraction)
        expected_result = 1 + int(fraction * prod([len(_) for _ in grid.values()]))
        self.assertEqual(result, expected_result)


class TestGetTopModelsFunction(unittest.TestCase):
    def test_get_top_models(self):

        grid_search_mock = Mock()

        grid_search_mock.cv_results_ = {
                "rank_test_score": [1, 2, 3],
                "mean_test_score": [0.9, 0.85, 0.8],
                "std_test_score": [0.05, 0.02, 0.1],
                "params": [{"param1": 1}, {"param1": 2}, {"param1": 3}],
        }

        top_results = get_top_models(grid_search_mock, top=2)

        expected_results = [
            (1, 0.9, 0.05, {"param1": 1}),
            (2, 0.85, 0.02, {"param1": 2})
        ]

        self.assertEqual(top_results, expected_results)


# DATASETS.PY
class TestMyModule(unittest.TestCase):
    def test_list_available_datasets(self):
        available_datasets = list_available_datasets()
        self.assertIsInstance(available_datasets, tuple)
    
    def test_get_data(self):
        _available_datasets = ['flchain', 'gbsg2', 'metabric', 'support']
        available_dataset = _available_datasets[0]
        dataset_object = get_data(available_dataset)
        self.assertIsInstance(dataset_object, dataset)
        
        unavailable_dataset = "non_esistente"
        with self.assertRaises(NameError):
            get_data(unavailable_dataset)
    
    def test_dataset_methods(self):
        data = {"event": [1, 0, 1], "time": [10, 20, 15], "feature": [0.1, 0.5, 0.3]}
        df = pd.DataFrame(data)
        my_dataset = dataset(name="test_dataset", dataframe=df)
        
        X, y = my_dataset.get_X_y()
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, np.ndarray)


# ADAPTER.PY
class TestSurvivalEstimator(unittest.TestCase):
    def test_fit_predict(self):
        estimator = SurvivalEstimator()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([(True, 100), (False, 200)], dtype=[('event', bool), ('time', int)])
        
        estimator.fit(X, y)
        
        predictions = estimator.predict(X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, y.shape)
        self.assertTrue(np.all(predictions == 1))
    
    def test_harrell_score(self):
        estimator = SurvivalEstimator()
        y_true = np.array([(True, 1), (False, 2)], dtype=[('event', bool), ('time', int)])
        y_pred = np.array([1.0, 2.0])
        
        score = estimator.harrell_score(y_true, y_pred)
        self.assertIsInstance(score, tuple)
    
    def test_score(self):
        estimator = SurvivalEstimator()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([(True, 1000), (False, 200)], dtype=[('event', bool), ('time', int)])
        
        score = estimator.score(X, y)
        self.assertIsInstance(score, float)


# AUTON_ADAPTERS.PY
class TestDeepSurvivalMachines(unittest.TestCase):
    def setUp(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([(True, 1), (False, 2)], dtype=[('event', bool), ('time', int)])
        self.X = X
        self.y = y

    def test_fit(self):
        model = DeepSurvivalMachines()
        model.fit(self.X, self.y)
        self.assertIsNotNone(model.model_)
        self.assertTrue(hasattr(model, 'median_time_'))

    def test_predict_single_event(self):
        model = DeepSurvivalMachines()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],))

    def test_predict_multiple_events(self):
        model = DeepSurvivalMachines()
        model.fit(self.X, self.y)
        eval_times = np.array([1, 2, 3])
        predictions = model.predict(self.X, eval_times=eval_times)
        self.assertEqual(predictions.shape, (self.X.shape[0], eval_times.shape[0]))

    def test_get_parameter_grid(self):
        max_width = 10
        parameter_grid = DeepSurvivalMachines.get_parameter_grid(max_width)
        self.assertIsInstance(parameter_grid, dict)
        self.assertIn('n_distr', parameter_grid)
        self.assertIn('distr_kind', parameter_grid)
        self.assertIn('batch_size', parameter_grid)
        self.assertIn('layer_sizes', parameter_grid)
        self.assertIn('learning_rate', parameter_grid)
        self.assertIn('validation_size', parameter_grid)
        self.assertIn('max_epochs', parameter_grid)
        self.assertIn('elbo', parameter_grid)

# PYCOX_ADAPTERS.PY
class TestDeepHitSingle(unittest.TestCase):

    def setUp(self):
        self.deephit = DeepHitSingle(num_durations=10, layer_sizes=[10, 10], epochs=10, batch_size=16, validation_size=0.1)

    def test_fit_predict(self):
        X_train = np.random.rand(100, 5)
        y_train = np.array([(bool(random.getrandbits(1)),ele) for ele in random.sample(range(1000), 100)], dtype=[('event', bool), ('time', int)])
        X_test = np.random.rand(10, 5)

        self.assertIsNone(self.deephit.fit(X_train, y_train))

        eval_times = [10, 20, 30]
        predictions = self.deephit.predict(X_test, eval_times)

        self.assertTrue(all(0 <= p <= 1 for p in predictions))

        self.assertEqual(len(eval_times) * X_test.shape[0], len(predictions))


# UTIL.PY
class TestSurvivalFunctions(unittest.TestCase):
    def test_load_test_data(self):
        X, y = load_test_data()
        self.assertEqual(X.shape[0], y.shape[0])

    def test_get_indicator(self):
        y = np.array([(True, 1), (False, 2)], dtype=[('censor', '?'), ('time', 'f8')])
        indicator = get_indicator(y)
        self.assertTrue(np.array_equal(indicator, [True, False]))

    def test_get_time(self):
        y = np.array([(True, 1), (False, 2)], dtype=[('censor', '?'), ('time', 'f8')])
        time = get_time(y)
        self.assertTrue(np.array_equal(time, [1, 2]))

    def test_survival_train_test_split(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([(True, 1), (True, 2), (False, 3), (False, 4)], dtype=[('censor', '?'), ('time', 'f8')])
        X_train, X_test, y_train, y_test = survival_train_test_split(X, y, test_size=0.50, rng_seed=42)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], X.shape[0])
        self.assertEqual(y_train.shape[0] + y_test.shape[0], y.shape[0])

    def test_survival_crossval_splitter(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([(True, 1), (True, 2), (False, 3), (False, 4), (True, 5)], dtype=[('censor', '?'), ('time', 'f8')])
        splitter = survival_crossval_splitter(X, y, n_splits=2, n_repeats=1, rng_seed=42)
        self.assertFalse(isinstance(splitter, RepeatedStratifiedKFold))

if __name__ == "__main__":
    unittest.main()
