# Docker + Prometheus + Grafana — Setup

Run these from `docker/` (i.e. `D:\surge-drift-monitor\docker`) unless
noted otherwise. This is a SEPARATE compose project from your existing
`docker/mlflow/docker-compose.yml` — the two are independent; you can run
either, both, or neither, and starting/stopping one has no effect on the
other.

**Port note:** your `docker/mlflow` service and the `mlflow server`
command you've been running directly on Windows both use port 5000 —
don't run both at once. Keep using the local `mlflow server` process for
now (everything in Phase 3/4 was tested against that); the containerized
`mlflow` service can stay parked/unused unless you deliberately switch to it.

Order matters for the steps below: the historical backfill must happen
BEFORE Prometheus's container starts, since it writes TSDB blocks directly
into the storage directory Prometheus reads on startup.

## 1. Generate the OpenMetrics backfill file

Run from the project root (`D:\surge-drift-monitor`), not from `docker/`:

```powershell
python -m src.monitoring.backfill_prometheus --output docker/prometheus/backfill.openmetrics
```

Reads your existing `metrics_timeseries.parquet` + `retrain_events.jsonl`
(from whichever backtest run you want to visualize) and converts them into
a file Prometheus's backfill tool can load.

## 2. Load the backfill into Prometheus's storage (one-off, before first start)

`promtool` ships inside the official Prometheus image, so this runs it via
a throwaway container — no local install needed.

```powershell
cd docker\prometheus
docker run --rm -v "${PWD}:/work" prom/prometheus:latest promtool tsdb create-blocks-from openmetrics /work/backfill.openmetrics /work/data
cd ..\..
```

This creates block subdirectories under `docker/prometheus/data/` — the
same path `docker/docker-compose.yml` mounts as Prometheus's storage. If
this step reports errors about invalid samples, don't proceed to step 3 --
paste the error here first.

## 3. Start the stack

```powershell
cd docker
docker compose up -d --build
```

This starts `prometheus`, `exporter`, and `grafana` (three services, this
compose file only — your separate `docker/mlflow` project is untouched by
this command). First run pulls the Prometheus and Grafana images and
builds the exporter — expect a couple of minutes. Subsequent starts are fast.

## 4. Open Grafana

**http://localhost:3000** — login `admin` / `admin` (it'll prompt to change
the password; safe to skip for local use). The "Surge-Price Drift Monitor"
dashboard is already provisioned and should appear on the home page —
default time range is set to the full Jul–Dec 2024 backtest period, so the
whole drift ramp / trigger / retrain-attempt story should be visible
immediately, not just "no data" until you manually adjust the time picker.

## 5. Check Prometheus directly (optional, for debugging)

**http://localhost:9090** — use the "Graph" tab to query e.g. `surge_psi`
or `surge_retrain_trigger_fired` directly if a Grafana panel looks empty,
to check whether the problem is missing data vs. a panel query issue.

## Re-running the backfill later

If you rerun a backtest and want Grafana to reflect the new results:

```powershell
cd docker
docker compose stop prometheus
Remove-Item -Recurse -Force prometheus\data
cd ..
python -m src.monitoring.backfill_prometheus --output docker/prometheus/backfill.openmetrics
cd docker\prometheus
docker run --rm -v "${PWD}:/work" prom/prometheus:latest promtool tsdb create-blocks-from openmetrics /work/backfill.openmetrics /work/data
cd ..
docker compose up -d prometheus
cd ..
```

Deleting `docker/prometheus/data` first matters — re-running the backfill
into an existing data directory with overlapping timestamps can error or
produce duplicate blocks.

## Stopping everything

```powershell
cd docker
docker compose down
```

Add `-v` to also delete the Grafana settings volume (dashboard stays
provisioned either way, since that's file-based, not stored in the volume).
This does NOT touch your separate `docker/mlflow` project or its volumes.