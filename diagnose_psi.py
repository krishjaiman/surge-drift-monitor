import json
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# --- 1. Which sub-metrics are actually firing? ---
df = pd.read_parquet("data/monitoring/metrics_timeseries.parquet")
print("=== Breach rates by category ===")
print("drift_breached rate:", df["drift_breached"].mean())
print("performance_degraded rate:", df["performance_degraded"].mean())
print("zones_with_bias_alert - mean/max:", df["zones_with_bias_alert"].mean(), df["zones_with_bias_alert"].max())
print("zones_with_calibration_alert - mean/max:", df["zones_with_calibration_alert"].mean(), df["zones_with_calibration_alert"].max())
print("prediction_variance_breached rate:", df["prediction_variance_breached"].mean())
print()

# --- 2. Full PSI stats, no truncation ---
print("=== Full PSI stats (all 10 features) ===")
psi_cols = [c for c in df.columns if c.startswith("psi__")]
print(df[psi_cols].mean().sort_values(ascending=False))
print()

# --- 3. Reference snapshot bin structure for the suspect low-cardinality features ---
with open("data/reference/reference_snapshot_current.json") as f:
    snap = json.load(f)

for feat in ["day_of_week", "hour_of_day", "zone_id"]:
    profile = snap["features"][feat]
    print(f"=== reference profile: {feat} ===")
    print("dtype:", profile["dtype"])
    if profile["dtype"] == "numeric":
        print("bin_edges:", profile["histogram"]["bin_edges"])
        print("counts:", profile["histogram"]["counts"])
    else:
        print("category_frequencies:", profile["category_frequencies"])
    print()

# --- 4. Actual current values for these features, straight from the prediction log ---
from src.monitoring.prediction_log_reader import load_predictions
current = load_predictions("data/predictions", pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-02"))
for feat in ["day_of_week", "hour_of_day", "zone_id"]:
    print(f"=== current data: {feat} ===")
    print("dtype:", current[feat].dtype)
    print("sample values:", current[feat].unique()[:15])
    print()