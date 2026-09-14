# Aircraft RUL Predictor

A condition-aware remaining-useful-life prediction pipeline for the NASA C-MAPSS turbofan-engine dataset.

The final model uses an LSTM to estimate capped RUL from the latest 30 operating cycles. FD004 is the primary experiment because it contains six operating conditions and two fault modes.

## Problem

The training trajectories continue until engine failure. Each official test trajectory ends before failure, and the goal is to predict how many operational cycles remain after its final observation.

RUL targets are capped at 125 cycles so the model focuses on the degradation region rather than distinguishing between very large early-life values.

## Main engineering decisions

- Split development data by engine, not by row, to prevent overlapping sequences from the same engine leaking into training and validation.
- Fit all scalers, operating-condition clustering and sensor statistics using training data only.
- Discover six operating conditions from the three operational settings using KMeans.
- Normalize sensors separately within each operating condition.
- Use 30-cycle sequences containing 21 normalized sensors and six one-hot condition features.
- Penalize RUL overprediction more heavily, especially when actual RUL is at most 30 cycles.
- Verify the loss improvement across five paired random-seed experiments.
- Select the model using validation data before evaluating the official test set.

## Final model

```text
Architecture:                     LSTM
Sequence length:                  30 cycles
Features per timestep:            27
Hidden size:                      64
LSTM layers:                      1
RUL cap:                          125
General overprediction weight:    2
Critical RUL threshold:           30
Critical overprediction weight:   5
```

## Official FD004 test results

Metrics against capped RUL:

| Model | MAE | RMSE |
|---|---:|---:|
| Dummy median | 40.79 | 52.50 |
| Random Forest, final timestep | 14.38 | 19.63 |
| Boosted tree, final timestep | 14.34 | 19.43 |
| Safety-aware LSTM, 30 timesteps | **10.49** | **14.99** |

The tree baselines receive only the final observation. The LSTM receives the complete 30-cycle history.

Metrics against uncapped official RUL:

| Model | MAE | RMSE | C-MAPSS score |
|---|---:|---:|---:|
| Baseline asymmetric LSTM | 19.74 | 28.27 | **7263.72** |
| Safety-aware LSTM | **19.18** | **27.70** | 7307.85 |
| Random Forest | 23.07 | 30.82 | 8289.74 |
| Boosted tree | 23.03 | 30.51 | 8546.62 |

The safety-aware model improved MAE and RMSE and produced a better per-engine C-MAPSS score for 138 of 248 test engines. The baseline LSTM retained a marginal 0.61% advantage in total C-MAPSS score because the exponential score is strongly affected by a small number of large errors.

## Validation evidence

Across five paired random seeds, the safety-aware loss:

- reduced critical-region overprediction rate in 5/5 runs;
- reduced mean critical overprediction in 5/5 runs;
- reduced severe critical overprediction in 5/5 runs;
- improved overall and critical MAE/RMSE in 4/5 runs.

The official test set was not used for further model tuning.

## Pipeline

```text
Raw engine history
→ validate and sort cycles
→ scale operating settings
→ assign operating condition
→ condition-wise sensor normalization
→ one-hot condition encoding
→ select final 30 cycles
→ LSTM inference
→ capped RUL prediction
```

The saved inference pipeline includes the fitted setting scaler, KMeans model, condition-specific sensor statistics, feature order and LSTM parameters.

## Setup

This project uses Python 3.14 and `uv`.

```bash
git clone <repository-url>
cd aircraft-rul-predictor
uv sync
```

Run the tests:

```bash
uv run pytest -v
```

Start Jupyter:

```bash
uv run jupyter lab
```

## Command-line inference

The input CSV must contain raw rows for exactly one engine, including:

```text
engine_id
cycle
setting_1 ... setting_3
sensor_1 ... sensor_21
```

Run:

```bash
uv run aircraft-rul-predictor \
    --input data/examples/fd004_engine_1_history.csv
```

Example output:

```json
{
    "engine_id": 1,
    "ending_cycle": 230,
    "ending_condition": 2,
    "timesteps_received": 230,
    "padded_timesteps": 0,
    "raw_predicted_rul": 24.8689,
    "predicted_rul": 24.8689,
    "rul_cap": 125.0
}
```

## Repository structure

```text
data/raw/          Original C-MAPSS files
data/processed/    Final test predictions
data/examples/     Example inference input
models/fd004/      Model checkpoints and fitted preprocessing
notebooks/         Experiments and analysis
reports/           Metrics and comparison tables
src/               Reusable inference pipeline
tests/             Automated model and pipeline tests
```

## Reproducibility

The repository records:

- fixed random seeds;
- engine-level data splits;
- training and loss configuration;
- five-seed robustness results;
- fitted preprocessing artifacts;
- final test predictions;
- per-engine C-MAPSS score contributions.

## Limitations

- C-MAPSS contains simulated rather than real aircraft operations.
- RUL above 125 cycles is intentionally collapsed into one target value.
- Operating conditions are approximated using KMeans.
- The model does not quantify predictive uncertainty.
- The system is a portfolio demonstration, not a safety-certified maintenance system.