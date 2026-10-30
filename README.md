# Day-ahead electricity demand forecasting

Predict the 48 half-hours of a target day from the seven preceding daily profiles and the target weekday: **343 features**, with errors reported in **MW**.

## Consistent comparison

All models share the same data and five chronological validation windows. Training expands at each window; each validation period contains 748 days. The former training and validation partitions form the development set (4,493 days, January 8, 2012 through April 26, 2024). The test set is excluded from cross-validation and evaluated separately below.

| Window | Training days | Validation start | Validation end |
|---|---:|---|---|
| 1 | 753 | 2014-01-30 | 2016-02-16 |
| 2 | 1,501 | 2016-02-17 | 2018-03-05 |
| 3 | 2,249 | 2018-03-06 | 2020-03-22 |
| 4 | 2,997 | 2020-03-23 | 2022-04-09 |
| 5 | 3,745 | 2022-04-10 | 2024-04-26 |

- **D−7 persistence**: main baseline, repeats the previous week's profile.
- **D−1 persistence**: additional reference, repeats the previous day's profile.
- **Linear**: multi-output linear regression with standardized inputs.
- **Ridge α=10**: the same regression with L2 regularization.
- **NN**: dense network 64 → 32 → 48, ReLU, Adam at 0.0003, standardized inputs and targets, random seed 42.

Each model is recreated for each window. Scalers use training data only. The NN selects its epoch count on the last 15% of the training window, with a maximum of 120 epochs and patience of 12. Inner scalers use only the first 85%. The network is then reinitialized and refitted on the entire training window for the selected number of epochs. Outer validation data never controls early stopping.

Forecasts are updated daily: observations through D−1 are available. Historical inputs can therefore overlap previously evaluated days. This is daily forecasting rather than a single forecast issued for several years ahead.

## Cross-validation results

| Model | MAE (MW) | RMSE (MW) | MAE gain against D−7 | RMSE gain against D−7 |
|---|---:|---:|---:|---:|
| Ridge α=10 | **969.35** | **1,480.35** | **72.24%** | **70.81%** |
| Linear | 1,153.25 | 1,689.66 | 66.98% | 66.69% |
| NN | 1,217.35 | 1,722.41 | 65.14% | 66.04% |
| D−1 persistence | 2,899.39 | 4,295.47 | 16.98% | 15.31% |
| D−7 persistence | 3,492.24 | 5,072.00 | 0% | 0% |

**Gain (%) = 100 × (1 − model error / D−7 error).** Window gains use that window's baseline. Overall metrics pool errors across all evaluated days: weighted mean MAE and square root of weighted MSE for RMSE. The standard deviation of window MAEs is also reported; it is not a confidence interval.

Ridge also reduces MAE by **66.57% against D−1**, the stronger persistence reference here. The NN does not improve on the linear models overall. Dates, scores, and NN epochs are in [cv_scores.csv](results/cv_scores.csv), the summary in [cv_summary.csv](results/cv_summary.csv), and charts in [the notebook](first_regression.ipynb) and [cv_comparison.png](results/cv_comparison.png).

These results are exploratory: α=10 was chosen after earlier experiments, without nested selection within the current windows. The NN uses one seed. Earlier evaluations with different features or partitions are not directly comparable. The test set was consulted previously; an untouched future period would be preferable for independent confirmation.

## Test-set reevaluation

After fitting on all 4,493 development days, fixed settings are evaluated on the **795 days from April 27, 2024 through June 30, 2026**. The NN selects 99 epochs using an internal development holdout, then refits on the full development set. No fitting decision uses test targets. Histories are updated daily, without retraining model parameters during the test period.

| Model | MAE (MW) | RMSE (MW) | MAE gain against D−7 | MAE gain against D−1 |
|---|---:|---:|---:|---:|
| Ridge α=10 | **946.32** | 1,392.21 | **70.95%** | **61.94%** |
| Linear | 953.72 | **1,379.03** | 70.72% | 61.64% |
| NN | 1,164.68 | 1,597.63 | 64.24% | 53.16% |
| D−1 persistence | 2,486.28 | 3,626.02 | 23.67% | 0% |
| D−7 persistence | 3,257.19 | 4,790.46 | 0% | −31.01% |

Ridge has the lowest MAE, but its margin over Linear is small: 7.39 MW, approximately 0.78%. Linear has the lowest RMSE. These figures alone do not establish statistical significance. Earlier consultation of the test set means this reevaluation is not a fresh independent assessment. Full results: [test_scores.csv](results/test_scores.csv).

## Data exploration

[Data_explore.ipynb](Data_explore.ipynb) includes data-quality checks, temporal evolution, weekday and seasonal profiles, month/hour seasonality, distributions, extreme days, and lags from 1 to 14 days. D−1 and D−7 persistence errors are also compared by target weekday on the same days.

Consumption charts use development data only. Quality checks cover the entire source: 5,295 complete days of 48 readings, with no missing days or duplicate Date/Time rows. The 50% raw missingness occurs at minutes 15 and 45; half-hourly readings are complete. Profiles are higher in winter and lower on weekends. D−1 persistence errors are largest on Mondays and Saturdays.

Exported charts and tables are in [results/eda](results/eda/). These descriptive statistics do not replace temporal validation scores. The source CSV retains the provider's original headers; [the ingestion mapping](demand_forecasting/data.py) exposes English column names in code and outputs.

## Running the project

```bash
python -m pip install -r requirements.txt
python data_processing.py
python first_regression.py
python first_regression.py --test
python -m unittest discover -s tests -v
```

[data_processing.py](data_processing.py) checks consecutive days, 48 readings per day, and duplicate timestamps, then creates the shared dataset with dates. Original chronological split boundaries are preserved. The older one-day datasets are no longer generated or used by the benchmark.

[first_regression.py](first_regression.py) runs the 25 cross-validation evaluations and exports both CSV files. Use `--test` for explicit test evaluation. [first_regression.ipynb](first_regression.ipynb) displays saved results by default; set `RECOMPUTE = True` to rerun training and update tables and charts. After changing data or models, recompute before interpreting saved outputs.
