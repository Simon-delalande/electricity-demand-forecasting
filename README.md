## Baseline models and evaluation strategy

The first stage of this project establishes a simple forecasting benchmark before introducing more complex models. Each input contains the 48 half-hourly consumption values from one day, and the target contains the corresponding values for the following day.

Two approaches are compared:

- **Persistence baseline:** use the previous day's consumption profile as the forecast for the next day.
- **Linear regression:** learn an affine mapping from the previous day's profile to the next day's profile.

The linear regression uses standardized inputs, with scaling parameters fitted exclusively on the training data. Targets remain in their original units, so prediction errors retain their physical interpretation.

### Time-series cross-validation

To assess whether the improvement holds across different periods, both approaches are evaluated using five chronological validation windows with an expanding training set.

For each window, the scaler and regression are fitted using only preceding observations. Both models are then evaluated on the same validation period. The final test set is excluded from this procedure.

Performance is measured by the **mean absolute error (MAE)**, averaged across all days and half-hourly intervals in each validation window.

| Validation window | Persistence MAE | Linear regression MAE | MAE reduction |
|---|---:|---:|---:|
| 1 | 3,158.11 | 1,721.43 | 45.49% |
| 2 | 3,090.07 | 1,610.09 | 47.89% |
| 3 | 2,978.81 | 1,417.74 | 52.41% |
| 4 | 3,046.47 | 1,894.09 | 37.83% |
| 5 | 2,751.19 | 1,776.42 | 35.43% |

MAE reduction is calculated relative to the persistence baseline:

\[
\text{MAE reduction}
=
100 \times \left(
1-\frac{\text{MAE}_{\text{linear regression}}}
{\text{MAE}_{\text{persistence}}}
\right).
\]

### Findings and next steps

Linear regression outperforms persistence in all five validation windows, reducing MAE by **35.4% to 52.4%**. The arithmetic mean of the five window-level reductions is **43.8%**.

These results show that the improvement extends across several evaluation periods. Its magnitude varies, with smaller gains in the last two windows. Further analysis will examine errors by time of day and calendar period to understand this variation.

This establishes a benchmark for evaluating additional features and more complex models. Final performance will be assessed on the held-out test set after model selection.
