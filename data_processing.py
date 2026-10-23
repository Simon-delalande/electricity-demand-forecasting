#Our first goal is to make a prediction for the 48 half-hours of
# the next day by using today's consumption 

import pandas as pd
import numpy as np
from pathlib import Path


data_dir = Path(__file__).resolve().parent / "Data"
df = pd.read_csv(
    data_dir / "data.csv", sep=";",
    usecols=["Date", "Heure", "Consommation (MW)"],
)
df["Date"] = pd.to_datetime(df["Date"])

#The dataframe is sorted to avoid training on future data
df = df.sort_values(["Date", "Heure"])

#We remove the nans for the target value
df = df.dropna(subset=["Consommation (MW)"]).copy()

data = (
    df.groupby("Date")["Consommation (MW)"]
      .agg(list)
      .reset_index(name="Consumption")
)




conso = data["Consumption"]
#lag 1 is the data from yesterday, on which we do the regression to predict today's data
data["lag_1"] = conso.shift(1)
# Encode the target day's weekday: Monday=0, ..., Sunday=6.
weekday_one_hot = np.eye(7, dtype=np.uint8)[data["Date"].dt.dayofweek]
data= data.iloc[1:]
weekday_one_hot = weekday_one_hot[1:]
data= data.drop(["Date"], axis=1)

n = len(data)
cut_train = int(0.70 * n)
cut_val = int(0.85 * n)

data_train = data.iloc[:cut_train].copy()
data_val = data.iloc[cut_train:cut_val].copy()
data_test = data.iloc[cut_val:].copy()

target = "Consumption"

X_train = np.stack(data_train["lag_1"].to_numpy())
y_train = np.stack(data_train["Consumption"].to_numpy())

X_val = np.stack(data_val["lag_1"].to_numpy())
y_val = np.stack(data_val["Consumption"].to_numpy())

X_test = np.stack(data_test["lag_1"].to_numpy())
y_test = np.stack(data_test["Consumption"].to_numpy())

np.savez_compressed(
    data_dir / "dataset.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

# Keep the same samples, chronological splits and targets as the baseline.
# Input columns: 48 consumption values, then Monday through Sunday.
np.savez_compressed(
    data_dir / "dataset_weekday.npz",
    X_train=np.concatenate([X_train, weekday_one_hot[:cut_train]], axis=1),
    y_train=y_train,
    X_val=np.concatenate([X_val, weekday_one_hot[cut_train:cut_val]], axis=1),
    y_val=y_val,
    X_test=np.concatenate([X_test, weekday_one_hot[cut_val:]], axis=1),
    y_test=y_test,
)
