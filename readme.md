# Surge-Price Drift Monitor & Automated Retraining Orchestrator

A production-style MLOps system that trains a demand forecasting model, serves it,
continuously monitors it for drift, and **autonomously retrains and promotes a new
model when the old one degrades** — without human intervention. Built to demonstrate
end-to-end ML system design skills for an MLE/MLOps role, using real NYC TLC Yellow
Taxi trip data as a stand-in for a ride-hailing demand-forecasting pipeline.

## Problem Statement

Production ML models degrade silently. A demand forecaster feeding a surge-pricing
engine can drift for days before anyone notices — covariate drift (rider behavior,
seasonality), concept drift (new transit lines, remote work), or plain staleness from
a model that was trained once and never touched again. By the time a human notices
via downstream KPI regressions, the damage (mispriced surges, mispositioned drivers)
is already done.

This project builds the missing piece: a system that detects that degradation
automatically, decides whether it's real, and fixes it — retraining and promoting a
new model without a person in the loop, then re-baselining monitoring so the cycle
can repeat.

## What It Actually Does (Pipeline)

```
Train → Serve → Monitor → Detect Drift → Retrain → Compare → Promote → Reset Baseline
  ↑___________________________________________________________________________|
```

1. **Trains** a LightGBM demand model on historical NYC taxi trip data, freezing a
   statistical "reference snapshot" of every input feature at training time.
2. **Serves** predictions via FastAPI, logging every prediction and its inputs.
3. **Monitors** live predictions against the frozen reference snapshot hourly: PSI,
   label KL divergence, null-rate drift, and feature-prediction correlation stability
   for covariate/concept drift; rolling RMSE/MAE, per-zone bias, calibration error,
   and prediction variance for performance drift.
4. **Triggers** a retrain only when **both** drift AND performance degrade
   simultaneously (drift alone is treated as insufficient evidence).
5. **Retrains** a challenger on the most recent 30 days, evaluates it against the
   current champion on a held-out slice, and **only promotes it if it beats the
   champion by >5% RMSE** — a hard safety gate against regressions.
6. On promotion, **rebuilds the reference snapshot** and **resets monitoring state**
   so drift detection starts clean against the new baseline.
7. Everything is **visualized in Grafana** via Prometheus, including a full
   historical backfill of the 6-month backtest.

## Results (backtested against 6 months of real production-like data, Jul–Dec 2024)

- **4,416-hour (6-month) historical backtest** replaying real NYC taxi demand across
  **261 zones**, comparing live predictions against a model frozen at training time.
- **Drift detection false-positive rate ≈ 9.8%** measured on the pre-drift "healthy"
  period — under the project's <10% target — while correctly ramping to sustained
  trigger firing once the frozen model genuinely went stale (validated by comparing
  early-period vs. late-period trigger rates, not just an aggregate).
- **10 features monitored for covariate/concept drift** (PSI, correlation stability,
  null-rate) and **5 performance dimensions per zone** (RMSE/MAE across 3 rolling
  windows, bias, calibration error, prediction variance).
- **Two known, documented measurement limitations**, not silently hidden: PSI on
  temporally-autocorrelated weather features is structurally noisy under any
  practical window size (excluded from the trigger, still logged for visibility);
  low-volume zones (<5 trips/hour) are excluded from percentage-based bias/
  calibration checks to avoid noise-driven false positives.
- **Autonomous retraining cooldown logic cut wasted compute ~23x** (683 → 29 retrain
  attempts across a 1,080-hour test window) by adding a failed-attempt cooldown
  alongside the promotion cooldown.
- **Champion-challenger safety gate correctly rejected 29/29 candidate models** that
  underperformed the incumbent (11–13% worse RMSE on held-out data), preventing 29
  potential production regressions — a real, measured demonstration of the
  promotion gate's core safety guarantee, not a hypothetical.
- **Full observability stack**: Prometheus + Grafana, containerized, with a
  6-panel live dashboard and a historical backfill pipeline (via
  `promtool tsdb create-blocks-from-openmetrics`) so the entire 6-month drift story
  is visible on first dashboard load, not just data collected after deployment.

## Tech Stack

| Component | Tool |
|---|---|
| Model | LightGBM |
| Experiment Tracking / Registry | MLflow (alias-based champion/challenger promotion) |
| Serving | FastAPI |
| Drift Detection | Custom (PSI, KL divergence, Spearman stability) — no external drift library |
| Data Versioning | DVC |
| Containerization | Docker / Docker Compose |
| Metrics & Dashboards | Prometheus + Grafana |
| Language | Python 3.11 |

## Project Structure

```
surge-drift-monitor/
├── src/
│   ├── training/         # Phase 1: model training + reference snapshot
│   ├── serving/           # Phase 2: FastAPI serving app
│   ├── monitoring/        # Phase 3: drift detection, alerting, Prometheus backfill
│   └── retraining/        # Phase 4: autonomous champion-challenger retraining
├── tests/                 # pytest suite across all phases
├── docker/
│   ├── prometheus/        # Prometheus config + backfilled historical data
│   ├── grafana/           # Provisioned datasource + dashboard
│   ├── exporter/          # Live metrics exporter (staging layer → Prometheus)
│   ├── mlflow/            # Containerized MLflow (optional; project also runs
│   │                         MLflow locally via `mlflow server`)
│   └── docker-compose.yml # Prometheus + exporter + Grafana
├── data/
│   ├── raw/                # DVC-tracked raw trip data
│   ├── processed/          # Feature-engineered training data
│   ├── reference/          # Frozen reference distribution snapshots
│   ├── predictions/        # Partitioned prediction log (Phase 2 output)
│   └── monitoring/         # Drift metrics time series + alert log (Phase 3/4 output)
├── params.yaml             # All thresholds, windows, and config — nothing hardcoded
└── requirements.txt
```

## Setup — Cloning and Running This Project

### Prerequisites
- Python 3.11+
- Docker Desktop (for the Prometheus/Grafana stack)
- ~2 GB free disk space for raw data + Docker images

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/surge-drift-monitor.git
cd surge-drift-monitor
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 2. Pull the data (DVC)

```bash
dvc pull
```

### 3. Start MLflow (local tracking server)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
Leave this running in its own terminal.

### 4. Train the baseline model (Phase 1)

```bash
python -m src.training.train
python -m src.training.reference_snapshot
```

### 5. Run the production replay (Phase 2)

Serves the model and replays held-out data hour-by-hour, logging predictions.
```bash
python -m src.serving.app        # in one terminal
python -m src.serving.replay_driver   # in another
```

### 6. Run the monitoring backtest (Phase 3)

```bash
python -m src.monitoring.backtest_runner
```

### 7. Run the integrated monitoring + retraining backtest (Phase 4)

```bash
python -m src.retraining.backtest_runner_with_retraining --start 2024-07-01T00:00:00 --end 2025-01-01T00:00:00
```

### 8. Stand up the observability stack

```bash
python -m src.monitoring.backfill_prometheus --output docker/prometheus/backfill.openmetrics
cd docker/prometheus
docker run --rm --entrypoint promtool -v "${PWD}:/work" prom/prometheus:latest tsdb create-blocks-from openmetrics /work/backfill.openmetrics /work/data
cd ..
docker compose up -d --build
```
Grafana: **http://localhost:3000** (admin/admin) — dashboard "Surge-Price Drift Monitor".
Prometheus: **http://localhost:9090**.

### 9. Run the test suite

```bash
pytest tests/ -v
```

## Known Limitations (documented, not hidden)

- **PSI on `temperature_2m`/`windspeed_10m`** is structurally elevated under any
  practical window size due to real-world temporal autocorrelation in weather data
  vs. a full-training-period reference distribution. These features are excluded
  from firing the retrain trigger (still computed and visible in Grafana). The
  principled fix — a seasonally-stratified reference baseline — is scoped out for
  now and documented as future work.
- **Zones averaging under 5 trips/hour** are excluded from percentage-based bias and
  calibration checks, since small-count percentage error is dominated by noise, not
  signal.
- **Champion-challenger retraining hyperparameters** are reused verbatim from the
  original (much larger, 2-year) training configuration; a 25–30 day retrain window
  may warrant separately-tuned hyperparameters — an open, documented question rather
  than a silently-accepted gap.

## License

MIT (or update as appropriate).