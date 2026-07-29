#!/usr/bin/env bash
# One-command setup for a fresh clone, per the project's reproducibility constraint.
# Usage: ./setup.sh
set -euo pipefail

echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/5] Starting MLflow tracking server (Docker)..."
(cd docker/mlflow && docker compose up -d --build)
echo "      MLflow UI available at http://localhost:5000"

echo "[4/5] Setting up DVC local remote (swap to GCS once you have a GCP account:"
echo "      dvc remote modify localremote url gs://<your-bucket>/dvcstore)"
mkdir -p /tmp/dvc-local-remote
dvc remote add -d localremote /tmp/dvc-local-remote --force

echo "[5/5] Running the Phase 1 pipeline (ingest -> features -> train -> reference snapshot)..."
dvc repro

echo ""
echo "Done. Champion model registered and aliased in MLflow at http://localhost:5000"
echo "Reference snapshot written to data/reference/reference_snapshot_current.json"
