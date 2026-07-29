# Phase 1 — Baseline

## What this delivers

- `src/ingestion/download_data.py` — downloads 6 sampled months of NYC TLC
  Yellow Taxi data, zone lookup, and NYC-wide weather. Schema-validates
  every download before it's trusted (fails loudly on upstream schema drift).
- `src/features/build_features.py` — aggregates raw trips into a
  zone-hour demand time series with time/lag/rolling/weather features.
  Zero-fills missing (zone, hour) combos so "no demand" is a real label,
  not a missing row. Lag/rolling features are strictly backward-looking
  (no target leakage).
- `src/training/train.py` — trains LightGBM, validates on a **date-based**
  split (not random — random splits leak autocorrelation between adjacent
  hours and give an overly optimistic RMSE). Logs to MLflow, registers the
  model, aliases it `@champion`. Also logs zone-level RMSE (worst + median)
  since global aggregates can hide zone-specific failures.
- `src/training/reference_snapshot.py` — the frozen baseline artifact.
  Computes per-feature stats (mean/std/percentiles/histogram) on the exact
  training data used by `train.py`. Everything in Phase 3 diffs against this.
- `dvc.yaml` + `params.yaml` — full pipeline definition, all tunables centralized.
- `tests/` — 18 unit tests covering the aggregation, lag-leakage, split, and
  snapshot logic. All passing.

## How to run

```bash
./setup.sh
```

This spins up MLflow in Docker, sets up a local DVC remote, and runs
`dvc repro` to execute the full pipeline: ingest → features → train → snapshot.

**Before running for real**: you have no GCP account yet, so the DVC remote
is a local folder (`/tmp/dvc-local-remote`) — fine for solo dev, but it means
your data isn't actually backed up anywhere durable yet. Get a GCP free-tier
account this week (you'll need it for Phase 4's Cloud Run deployment anyway),
then swap the remote with one command and no code changes:

```bash
gcloud auth application-default login
dvc remote modify localremote url gs://<your-bucket>/dvcstore
dvc push
```

## Verify it worked

```bash
# Run tests
pytest tests/ -v

# Check MLflow registry — you should see one registered model version
# aliased @champion
open http://localhost:5000

# Inspect the reference snapshot
python -c "import json; print(json.load(open('data/reference/reference_snapshot_current.json'))['metadata'])"
```

## Deliberately deferred (not bugs, just out of scope for Phase 1)

- **Zone-level weather** — currently one NYC-wide weather series joined by
  timestamp only. Per-zone weather would need per-zone lat/lon and N times
  the API calls; not worth it until the model is proven to need it.
- **US holiday calendar feature** — would meaningfully help demand
  prediction around holidays, skipped to avoid an extra dependency in
  Phase 1. Easy add later: one column, one `params.yaml` entry.
- **Hyperparameter tuning** — current LightGBM params in `params.yaml` are
  reasonable defaults, not tuned. Per your own project principle, Phase 1
  should be "done fast" — the project's value is in Phases 3/4, not
  squeezing out extra RMSE here.

## What I'd want you to look at before we move to Phase 2

1. `_assign_champion_alias()` in `train.py` unconditionally promotes —
   correct for Phase 1 (no existing champion to compare against), but
   flagging clearly so you don't copy this pattern into Phase 4, where it
   must be replaced by the champion-challenger gate.
2. The reference snapshot's `metadata.n_rows` — sanity check this against
   what you expect once you run on real (not synthetic) data with your
   6-month range.
