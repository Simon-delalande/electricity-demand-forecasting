"""Run the same chronological benchmark as first_regression.ipynb."""
import os
from pathlib import Path

# Keep CPU execution economical and reproducible on machines without a GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from data_processing import DATA_DIR, build_dataset
from demand_forecasting.evaluation import evaluate_holdout, evaluate_models, summarize_scores
from demand_forecasting.models import make_models

ROOT = Path(__file__).resolve().parent


def load_development_data():
    path = DATA_DIR / "dataset_weekday_7days.npz"
    if not path.exists():
        build_dataset()
    with np.load(path) as data:
        needs_dates = "dates_train" not in data.files
    if needs_dates:
        build_dataset()
    with np.load(path) as data:
        # The former validation partition joins development CV. Test stays reserved.
        return tuple(np.concatenate([data[f"{key}_train"], data[f"{key}_val"]])
                     for key in ["X", "y", "dates"])


def run_benchmark(nn_epochs=120, verbose=True):
    X, y, dates = load_development_data()
    with threadpool_limits(limits=1):
        scores = evaluate_models(make_models(nn_epochs), X, y, dates,
                                 TimeSeriesSplit(n_splits=5), verbose=verbose)
    return scores, summarize_scores(scores)


def run_test_benchmark(nn_epochs=120, verbose=True):
    X, y, dates = load_development_data()
    with np.load(DATA_DIR / "dataset_weekday_7days.npz") as data:
        X_test, y_test, dates_test = (data[f"{key}_test"] for key in ["X", "y", "dates"])
    with threadpool_limits(limits=1):
        return evaluate_holdout(make_models(nn_epochs), X, y, dates,
                                X_test, y_test, dates_test, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Evaluate the reserved test instead of CV.")
    args = parser.parse_args()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    if args.test:
        scores = run_test_benchmark()
        scores.to_csv(output / "test_scores.csv")
        print(scores.round(2).to_string())
    else:
        scores, summary = run_benchmark()
        scores.to_csv(output / "cv_scores.csv", index=False)
        summary.to_csv(output / "cv_summary.csv")
        print(summary.round(2).to_string())
