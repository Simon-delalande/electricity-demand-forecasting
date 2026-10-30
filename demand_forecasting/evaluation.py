"""Shared expanding-window evaluation; errors and gains stay in original MW."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASELINE = "Persistence D-7"


def evaluate_holdout(models, X_train, y_train, dates_train,
                     X_test, y_test, dates_test, verbose=True):
    """Fit on development only and score an explicitly requested later test."""
    for X, y, dates in [(X_train, y_train, dates_train), (X_test, y_test, dates_test)]:
        if not len(X) or len(X) != len(y) or len(y) != len(dates):
            raise ValueError("Features, targets and dates must have equal nonzero lengths.")
        if not np.all(dates[1:] > dates[:-1]):
            raise ValueError("Target dates must be strictly increasing.")
    if dates_train[-1] >= dates_test[0]:
        raise ValueError("Development must strictly precede the test set.")
    rows = []
    for name, estimator in models.items():
        if verbose:
            print(f"Test : {name}", flush=True)
        fitted = clone(estimator).fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        rows.append({
            "Model": name,
            "Training start": str(dates_train[0])[:10],
            "Training end": str(dates_train[-1])[:10],
            "Test start": str(dates_test[0])[:10],
            "Test end": str(dates_test[-1])[:10],
            "Training days": len(X_train), "Test days": len(X_test),
            "MAE (MW)": mean_absolute_error(y_test, prediction),
            "RMSE (MW)": np.sqrt(mean_squared_error(y_test, prediction)),
            "NN epochs": getattr(fitted, "best_epoch_", np.nan),
        })
    scores = pd.DataFrame(rows).set_index("Model")
    for lag in [7, 1]:
        for metric in ["MAE", "RMSE"]:
            baseline = scores.loc[f"Persistence D-{lag}", f"{metric} (MW)"]
            scores[f"Gain {metric} vs D-{lag} (%)"] = (
                100 * (1 - scores[f"{metric} (MW)"] / baseline)
                if baseline else np.nan)
    return scores.sort_values("MAE (MW)")


def evaluate_models(models, X, y, dates, cv, verbose=True):
    dates = np.asarray(dates)
    if len(X) != len(y) or len(y) != len(dates):
        raise ValueError("Features, targets and dates must have equal lengths.")
    if not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Target dates must be strictly increasing.")
    rows = []
    for window, (train, val) in enumerate(cv.split(X), start=1):
        if train[-1] >= val[0]:
            raise ValueError("Training must strictly precede evaluation.")
        for name, estimator in models.items():
            if verbose:
                print(f"Window {window}: {name}", flush=True)
            fitted = clone(estimator).fit(X[train], y[train])
            prediction = fitted.predict(X[val])
            rows.append({
                "Window": window, "Model": name,
                "Training start": str(dates[train[0]])[:10],
                "Training end": str(dates[train[-1]])[:10],
                "Validation start": str(dates[val[0]])[:10],
                "Validation end": str(dates[val[-1]])[:10],
                "Training days": len(train), "Validation days": len(val),
                "MAE (MW)": mean_absolute_error(y[val], prediction),
                "RMSE (MW)": np.sqrt(mean_squared_error(y[val], prediction)),
                "NN epochs": getattr(fitted, "best_epoch_", np.nan),
            })
    scores = pd.DataFrame(rows)
    reference = scores[scores["Model"] == BASELINE].set_index("Window")
    for metric in ["MAE", "RMSE"]:
        denominator = scores["Window"].map(reference[f"{metric} (MW)"])
        scores[f"Gain {metric} (%)"] = 100 * (
            1 - scores[f"{metric} (MW)"] / denominator.replace(0, np.nan))
    return scores


def summarize_scores(scores):
    rows = []
    for name, group in scores.groupby("Model", sort=False):
        weights = group["Validation days"]
        rows.append({
            "Model": name,
            "MAE (MW)": np.average(group["MAE (MW)"], weights=weights),
            "RMSE (MW)": np.sqrt(np.average(group["RMSE (MW)"] ** 2, weights=weights)),
            "MAE across-window std (MW)": group["MAE (MW)"].std(ddof=0),
        })
    summary = pd.DataFrame(rows).set_index("Model")
    for metric in ["MAE", "RMSE"]:
        baseline = summary.loc[BASELINE, f"{metric} (MW)"]
        summary[f"Gain {metric} (%)"] = (
            100 * (1 - summary[f"{metric} (MW)"] / baseline)
            if baseline else np.nan)
    return summary.sort_values("MAE (MW)")
