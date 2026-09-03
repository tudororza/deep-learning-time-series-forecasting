# Deep Learning Time-Series Forecasting

Bonus project for the TU Darmstadt Deep Learning course (Summer Semester 2026).
The objective is to forecast the hourly operational load for 96 time series and
beat the naive last-value baseline with a reproducible PyTorch model.

## Current status

- Downloaded benchmark files have been validated.
- Four dependency-free target-history baselines are implemented.
- A schema-valid validation prediction file has been generated.
- The deep-learning model and additional Jena weather experiment are pending.

## Baselines

`src/baselines.py` implements:

- last observed value;
- daily persistence (lag 24);
- weekly persistence (lag 168);
- the mean pattern of the four most recent weeks.

On an internal 336-hour holdout from the end of the public training data, the
scores were:

| Method | WAPE |
| --- | ---: |
| Last value | 0.545040 |
| Daily persistence | 0.418006 |
| Weekly persistence | 0.454876 |
| Four-week seasonal mean | 0.341780 |

This is a single internal temporal holdout, not the hidden leaderboard score.

## Data

Download the public benchmark from:

<https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026/tree/main>

Place these files in the repository root:

- `train.csv`
- `validation_input.csv`
- `forecast_index_validation.csv`
- `metadata.json`

The large CSV files are intentionally excluded from Git.

## Reproduce the baseline

Run the internal backtest:

```bash
python3 -m src.baselines backtest --train train.csv --horizon 336
```

Generate the four-week seasonal-mean submission:

```bash
python3 -m src.baselines predict \
  --train train.csv \
  --forecast-index forecast_index_validation.csv \
  --method weekly_mean \
  --output outputs/baseline_weekly_mean.csv
```

The generated CSV has the required columns:

```text
series_id,timestamp,prediction
```

## Repository layout

```text
src/baselines.py                  Baselines, WAPE, and prediction generation
outputs/baseline_weekly_mean.csv  First validation prediction
metadata.json                     Public benchmark contract
```

