# Deep Learning Time-Series Forecasting

TU Darmstadt Deep Learning bonus project (Summer Semester 2026). The project
forecasts hourly operational load for 96 anonymized series with a residual LSTM
and tests the same model family on Jena temperature and humidity forecasting.

## Current benchmark results

The chronological internal holdout is the final 336 hours of `train.csv`. It is
kept separate from fitting preprocessing statistics and model parameters.

| Method | Internal WAPE |
| --- | ---: |
| Last value | 0.545040 |
| Daily persistence (lag 24) | 0.418006 |
| Weekly persistence (lag 168) | 0.454876 |
| Four-week seasonal mean | 0.341780 |
| Target-history-only LSTM | 0.383264 |
| Full multivariate LSTM | **0.241293** |

The full model selected epoch 12. This internal score is used for model
selection; the public Hugging Face score must be reported separately because
its validation labels are hidden.

## Model

For each forecast block, the model reads 168 historical hours. A one-layer LSTM
with 64 hidden units encodes normalized target history and, in the full model,
the historical covariates. The final state is Layer-Normalized and combined with
an eight-dimensional learned series embedding and known future covariates. A
shared decoder produces 24 residual forecasts. These residuals are added to a
four-week seasonal mean, with weekly and last-value fallbacks when history is
shorter.

Continuous inputs are standardized using training-only statistics. Binary and
cyclic variables are left unchanged. Missing covariates receive a missingness
indicator, are forward-filled within their series, and finally use a
training-set median fallback. Batch Normalization is intentionally not used.

## Environment

Python 3.12 is used. To reproduce the tested CPU environment:

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
.venv/bin/pip install -r requirements-dev.txt
```

The training code runs a compute preflight and selects `cuda` only if allocation,
matrix multiplication, an LSTM forward/backward pass, and an optimizer update all
succeed. PyTorch exposes AMD ROCm devices through the `torch.cuda` API. Check the
current environment with:

```bash
.venv/bin/python -m scripts.gpu_preflight
```

On the development laptop, the Radeon 780M display driver is active, but the
current environment uses the CPU PyTorch wheel and does not have working ROCm
compute access. GPU training remains optional; inference is always CPU-safe.
Follow AMD's current Ryzen ROCm installation guide before installing a ROCm
PyTorch wheel. No kernel or operating-system change is required to reproduce the
CPU results.

## Benchmark data

Download the files from the [course dataset on Hugging Face](https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026/tree/main)
and place them in the repository root:

```text
train.csv
validation_input.csv
forecast_index_validation.csv
metadata.json
```

The large CSV files are intentionally excluded from Git.

Reproduce the four target-history baselines:

```bash
.venv/bin/python -m src.baselines backtest --train train.csv --horizon 336
```

Run a fast end-to-end model check:

```bash
.venv/bin/python train.py --config configs/benchmark_smoke.json
```

Train the ablation and full model:

```bash
.venv/bin/python train.py --config configs/benchmark_target_only.json
.venv/bin/python train.py --config configs/benchmark.json
```

Once the full model is selected, refit it on all public targets for its selected
number of epochs and generate `outputs/lstm_full.csv`:

```bash
.venv/bin/python train.py --config configs/benchmark.json --retrain-full
```

## Jena experiment

Download and extract the 2009-2016 Jena climate data:

```bash
.venv/bin/python -m scripts.download_jena
```

The downloaded archive used during development has SHA256:

```text
63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b
```

Ten-minute measurements are aggregated to hourly means. Invalid negative wind
speeds are treated as missing, wind direction is represented as x/y components,
and the targets are temperature `T (degC)` and relative humidity `rh (%)`.

```bash
.venv/bin/python train.py --config configs/jena_smoke.json
.venv/bin/python train.py --config configs/jena.json
```

The chronological split is 2009-2014 for training, 2015 for validation, and
2016 for final testing. The evaluation reports temperature/humidity MAE and
RMSE plus mean normalized MAE.

The selected epoch-12 LSTM obtained the following 2016 test results:

| Method | Temperature MAE | Humidity MAE | Mean normalized MAE |
| --- | ---: | ---: | ---: |
| Last value | 2.719 | 11.169 | 0.501 |
| Daily persistence | **2.432** | 9.884 | 0.445 |
| Weekly persistence | 4.565 | 12.157 | 0.639 |
| LSTM | 2.706 | **8.903** | **0.431** |

The LSTM has the best combined normalized error and humidity error, while daily
persistence remains slightly better for temperature alone.

## Private-evaluation archive

Build the archive from the selected, full-data checkpoint:

```bash
.venv/bin/python -m scripts.package_submission \
  --checkpoint checkpoints/benchmark_full.pt \
  --output dist/benchmark_model_submission.zip
```

The archive contains `predict.py`, `checkpoint.pt`, `requirements.txt`, the
required source modules, and this README. Inference is offline and CPU-safe. The
evaluator command is exactly:

```bash
python predict.py \
  --input_dir /data/input \
  --output_file /output/predictions.csv \
  --checkpoint /submission/checkpoint.pt
```

The inference code discovers the history, future-covariate, and forecast-index
CSVs by schema rather than depending on validation-specific filenames. Output is
validated to contain one finite, nonnegative prediction for every forecast-index
row in its original order.

Verify the packed archive in an isolated temporary directory:

```bash
.venv/bin/python -m scripts.verify_submission \
  --archive dist/benchmark_model_submission.zip \
  --input-dir .
```

## Tests

```bash
.venv/bin/pytest -q
```

Tests cover metrics, seasonal fallbacks, causal imputation, training-only
scaling, lazy window shapes, model forward/backward updates, 336-hour recursive
rollout, CPU/GPU agreement when a GPU is available, and the exact inference
contract.

## Repository structure

```text
configs/                 Reproducible benchmark and Jena experiment settings
scripts/                 GPU preflight, Jena download, and archive builder
src/data.py              Validation, preprocessing, and lazy windows
src/model.py             Residual LSTM architecture
src/training.py          Deterministic training and device-safe batching
src/benchmark.py         Benchmark evaluation and checkpoint construction
src/jena.py              Additional-dataset experiment
src/inference.py         Private-test input discovery and recursive forecasting
train.py                 Shared training entry point
predict.py               Required private-evaluation entry point
tests/                   Automated unit and integration tests
```

Report artifacts are in `report/`, the rendered PDF is
`output/pdf/final_report.pdf`, and report-ready comparison figures are in
`output/plots/`. Before submission, the three authors must replace the clearly
marked contribution placeholder with their agreed individual contributions and
add the public Hugging Face leaderboard score.
