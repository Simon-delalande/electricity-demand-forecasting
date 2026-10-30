"""Build the common day-ahead dataset, preserving the original split dates."""
from pathlib import Path

import numpy as np
import pandas as pd
from demand_forecasting.data import read_source_data

DATA_DIR = Path(__file__).resolve().parent / "Data"


def build_dataset(data_dir=DATA_DIR):
    df = read_source_data(data_dir / "data.csv",
                          columns=["Date", "Time", "Consumption (MW)"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Time"]).dropna(subset=["Consumption (MW)"])
    if df.duplicated(["Date", "Time"]).any():
        raise ValueError("Duplicate half-hour observations.")
    daily = df.groupby("Date")["Consumption (MW)"].agg(list)
    if not daily.map(len).eq(48).all():
        raise ValueError("Each day must contain exactly 48 half-hours.")
    if not daily.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily consumption must contain consecutive calendar days.")
    consumption = np.stack(daily.to_numpy())
    if not np.isfinite(consumption).all():
        raise ValueError("Consumption must be finite.")
    targets = np.arange(7, len(daily))
    weekdays = np.eye(7)[daily.index.dayofweek[targets]]
    X = np.concatenate([consumption[targets - lag] for lag in range(1, 8)]
                       + [weekdays], axis=1)
    y = consumption[targets]
    dates = daily.index.to_numpy(dtype="datetime64[D]")[targets]
    # Original 70/15/15 boundaries, before dropping six initial training days.
    n = len(daily) - 1
    positions = targets - 1
    masks = {
        "train": positions < int(0.70 * n),
        "val": (positions >= int(0.70 * n)) & (positions < int(0.85 * n)),
        "test": positions >= int(0.85 * n),
    }
    arrays = {}
    for split, mask in masks.items():
        arrays.update({f"X_{split}": X[mask], f"y_{split}": y[mask],
                       f"dates_{split}": dates[mask]})
    path = data_dir / "dataset_weekday_7days.npz"
    np.savez_compressed(path, **arrays)
    return path


if __name__ == "__main__":
    print(build_dataset())
