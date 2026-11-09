"""Unit tests for factory-anomaly-detector. Run with:
    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from factory_anomaly.generator import (
    generate_dataset, generate_stream, standardize, FEATURE_NAMES, FAULT_TYPES,
)
from factory_anomaly.detectors import IsolationForestDetector, AutoencoderDetector
from factory_anomaly.evaluate import (
    precision_recall_f1, per_fault_recall, confusion,
)


def _train_test(seed=1):
    X_train, _, _ = generate_dataset(n_normal=400, n_anomaly=0, seed=seed)
    X_test, y_test, faults = generate_dataset(
        n_normal=200, n_anomaly=120, seed=seed + 10)
    Xtr, Xte, _, _ = standardize(X_train, X_test)
    return Xtr, Xte, y_test, faults


class TestGenerator(unittest.TestCase):
    def test_dataset_shape_and_labels(self):
        X, y, faults = generate_dataset(n_normal=500, n_anomaly=150, seed=3)
        self.assertEqual(X.shape, (650, len(FEATURE_NAMES)))
        self.assertEqual(int(y.sum()), 150)
        self.assertEqual(int((y == 0).sum()), 500)
        # Every anomalous row carries a known fault name.
        self.assertTrue(set(faults[y == 1]) <= set(FAULT_TYPES))
        self.assertTrue(all(f == "" for f in faults[y == 0]))

    def test_deterministic_with_same_seed(self):
        X1, y1, _ = generate_dataset(seed=9)
        X2, y2, _ = generate_dataset(seed=9)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_faults_shift_features_up(self):
        # Fault rows should have clearly higher vibration / temperature
        # than normal rows on average.
        X, y, _ = generate_dataset(n_normal=800, n_anomaly=300, seed=5)
        self.assertGreater(X[y == 1, 0].mean(), X[y == 0, 0].mean() + 0.1)
        self.assertGreater(X[y == 1, 2].mean(), X[y == 0, 2].mean() + 1.0)

    def test_stream_fault_onset(self):
        X, y, faults = generate_stream(n_windows=200, fault_at=120,
                                        fault="overheating", seed=2)
        self.assertEqual(len(X), 200)
        self.assertEqual(int(y[:120].sum()), 0)
        self.assertEqual(int(y[120:].sum()), 80)
        self.assertTrue(all(f == "overheating" for f in faults[120:]))

    def test_standardize_uses_train_stats(self):
        Xtr = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
        Xte = np.array([[2.0, 20.0]])
        ztr, zte, mean, std = standardize(Xtr, Xte)
        np.testing.assert_allclose(mean, [2.0, 20.0])
        np.testing.assert_allclose(zte, [[0.0, 0.0]])
        np.testing.assert_allclose(ztr.mean(axis=0), [0.0, 0.0], atol=1e-12)


class TestDetectors(unittest.TestCase):
    def test_isolation_forest_beats_chance(self):
        Xtr, Xte, y, _ = _train_test()
        det = IsolationForestDetector(n_trees=50, seed=1).fit(Xtr)
        pred = det.predict(Xte)
        m = precision_recall_f1(y, pred)
        self.assertGreater(m["recall"], 0.70)
        self.assertGreater(m["precision"], 0.70)
        self.assertGreater(m["f1"], 0.70)
        # Scores in [0, 1], anomalies score higher on average.
        s = det.score_samples(Xte)
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        self.assertGreater(s[y == 1].mean(), s[y == 0].mean())

    def test_autoencoder_beats_chance(self):
        Xtr, Xte, y, _ = _train_test()
        det = AutoencoderDetector(epochs=200, seed=1).fit(Xtr)
        pred = det.predict(Xte)
        m = precision_recall_f1(y, pred)
        self.assertGreater(m["recall"], 0.70)
        self.assertGreater(m["precision"], 0.70)
        self.assertGreater(m["f1"], 0.70)
        s = det.score_samples(Xte)
        self.assertGreater(s[y == 1].mean(), s[y == 0].mean())

    def test_autoencoder_reconstruction_improves_with_training(self):
        Xtr, _, _, _ = _train_test()
        det = AutoencoderDetector(epochs=5, seed=1)
        det.fit(Xtr[:100])  # short fit just to check the plumbing runs
        # A longer fit on the same data should reconstruct it better.
        det_long = AutoencoderDetector(epochs=200, seed=1).fit(Xtr[:100])
        short_err = det.score_samples(Xtr[:100]).mean()
        long_err = det_long.score_samples(Xtr[:100]).mean()
        self.assertLess(long_err, short_err)

    def test_per_fault_recall_all_faults_detected(self):
        Xtr, Xte, y, faults = _train_test()
        for det in (IsolationForestDetector(n_trees=50, seed=1),
                    AutoencoderDetector(epochs=200, seed=1)):
            det.fit(Xtr)
            per_fault = per_fault_recall(y, det.predict(Xte), faults)
            for fault in FAULT_TYPES:
                self.assertGreater(per_fault[fault], 0.5,
                                   f"{type(det).__name__} missed {fault}")


class TestEvaluation(unittest.TestCase):
    def test_precision_recall_f1_known_case(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        m = precision_recall_f1(y_true, y_pred)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 1))

    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        m = precision_recall_f1(y, y)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual(confusion(y, y)["tn"], 2)


if __name__ == "__main__":
    unittest.main()
