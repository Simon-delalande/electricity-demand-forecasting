import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X, y, cv):
    rows = []

    for window, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        fitted_model = clone(model)
        fitted_model.fit(X[train_idx], y[train_idx])

        predictions = fitted_model.predict(X[val_idx])

        rows.append({
            "Fenêtre": window,
            "MAE": mean_absolute_error(y[val_idx], predictions),
            "RMSE": np.sqrt(
                mean_squared_error(y[val_idx], predictions)
            ),
        })

    return pd.DataFrame(rows).set_index("Fenêtre")