"""Checks for temporal isolation, lag selection and aggregate metrics."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit

from demand_forecasting.evaluation import evaluate_holdout, evaluate_models, summarize_scores
from demand_forecasting.models import PersistenceRegressor


class PastOnlyRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.last_seen_ = X[:, 0].max()
        return self

    def predict(self, X):
        assert np.all(X[:, 0] > self.last_seen_), "Future data used in training"
        return X[:, :2]


class EvaluationTests(unittest.TestCase):
    def test_holdout_is_later_and_never_used_in_fit(self):
        X = np.arange(280).reshape(20, 14)
        y = X[:, :2] + 1
        dates = np.arange("2020-01-01", "2020-01-21", dtype="datetime64[D]")
        models = {"Persistence D-7": PersistenceRegressor(),
                  "Persistence D-1": PersistenceRegressor(1),
                  "Past only": PastOnlyRegressor()}
        scores = evaluate_holdout(models, X[:15], y[:15], dates[:15],
                                  X[15:], y[15:], dates[15:], verbose=False)
        self.assertEqual(scores.loc["Past only", "MAE (MW)"], 1)
        self.assertTrue(scores["Test days"].eq(5).all())
        self.assertEqual(scores.loc["Persistence D-1", "Gain MAE vs D-1 (%)"], 0)
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            evaluate_holdout(models, X[:15], y[:15], dates[:15],
                             X[14:], y[14:], dates[14:], verbose=False)

    def test_persistence_extracts_profile_without_calendar(self):
        X = np.arange(3 * 343).reshape(3, 343)
        y = np.zeros((3, 48))
        for lag in [1, 7]:
            prediction = PersistenceRegressor(lag).fit(X, y).predict(X)
            np.testing.assert_array_equal(prediction, X[:, (lag-1)*48:lag*48])

    def test_all_models_share_progressive_windows_and_reference(self):
        X = np.arange(140).reshape(20, 7)
        y = X[:, :2] + 1
        dates = np.arange("2020-01-01", "2020-01-21", dtype="datetime64[D]")
        scores = evaluate_models(
            {"Persistence D-7": PersistenceRegressor(), "Past only": PastOnlyRegressor()},
            np.tile(X, (1, 2)), y, dates, TimeSeriesSplit(3), verbose=False)
        for _, group in scores.groupby("Window"):
            for col in ["Validation start", "Validation end", "Training days"]:
                self.assertEqual(group[col].nunique(), 1)
        self.assertTrue(scores[scores["Model"] == "Persistence D-7"]
                        ["Gain MAE (%)"].eq(0).all())
        summary = summarize_scores(scores)
        self.assertEqual(summary.loc["Past only", "MAE (MW)"], 1)

    def test_rmse_is_pooled_and_gains_use_aggregate_baseline(self):
        scores = pd.DataFrame({
            "Model": ["Persistence D-7"] * 2 + ["Linear"] * 2,
            "Validation days": [1, 3, 1, 3],
            "MAE (MW)": [2, 4, 1, 2], "RMSE (MW)": [2, 4, 1, 2],
        })
        summary = summarize_scores(scores)
        self.assertEqual(summary.loc["Linear", "MAE (MW)"], 1.75)
        self.assertAlmostEqual(summary.loc["Linear", "RMSE (MW)"], np.sqrt(3.25))
        self.assertEqual(summary.loc["Linear", "Gain MAE (%)"], 50)


if __name__ == "__main__":
    unittest.main()
