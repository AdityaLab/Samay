# Time-series Foundational Models Library Monorepo

Contains repos for the projects. Each repo is a seperate folder.

# Todo

## Moment

no zero-shot forecasting source code or checkpoints

## Model Profile

| Model Name | Task Functionality                                                     | Special Arguments |
| ---------- | ---------------------------------------------------------------------- | :---------------: |
| TimesFM    | Forecasting                                                            |         -         |
| Moment     | Forecasting<br />Imputation<br />Anomaly Detection<br />Classification | norm, mask_ratio |
| Chronos    | Forecasting                                                            |         -         |

All fintuning functions have general basic arguments "epoch" and "lr" with default values.
