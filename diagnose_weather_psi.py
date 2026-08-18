import json
import pandas as pd

pd.set_option("display.width", 200)

df = pd.read_parquet("data/monitoring/metrics_timeseries.parquet")
df["as_of"] = pd.to_datetime(df["as_of"])
df["month"] = df["as_of"].dt.to_period("M")

print("=== psi__temperature_2m by month ===")
print(df.groupby("month")["psi__temperature_2m"].mean())
print()
print("=== psi__windspeed_10m by month ===")
print(df.groupby("month")["psi__windspeed_10m"].mean())
print()

with open("data/reference/reference_snapshot_current.json") as f:
    snap = json.load(f)

for feat in ["temperature_2m", "windspeed_10m"]:
    profile = snap["features"][feat]
    print(f"=== reference profile: {feat} ===")
    print("mean/std/min/max:", profile["mean"], profile["std"], profile.get("min"), profile.get("max"))
    print("bin_edges:", profile["histogram"]["bin_edges"])
    print("counts:", profile["histogram"]["counts"])
    print()

# Current data range in an early-production week (July) vs a late one (December)
from src.monitoring.prediction_log_reader import load_predictions
july = load_predictions("data/predictions", pd.Timestamp("2024-07-08"), pd.Timestamp("2024-07-15"))
december = load_predictions("data/predictions", pd.Timestamp("2024-12-08"), pd.Timestamp("2024-12-15"))
for feat in ["temperature_2m", "windspeed_10m"]:
    print(f"=== current data range: {feat} ===")
    print("July week   - min/mean/max:", july[feat].min(), july[feat].mean(), july[feat].max())
    print("December wk - min/mean/max:", december[feat].min(), december[feat].mean(), december[feat].max())
    print()