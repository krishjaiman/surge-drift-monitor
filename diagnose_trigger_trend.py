import pandas as pd

df = pd.read_parquet("data/monitoring/metrics_timeseries.parquet")
df["as_of"] = pd.to_datetime(df["as_of"])
df["week"] = df["as_of"].dt.to_period("W")

print("=== Weekly retrain_trigger_fired rate (first 6 weeks vs. rest) ===")
weekly = df.groupby("week")["retrain_trigger_fired"].mean()
print(weekly)
print()
print("First 4 weeks avg trigger rate (proxy for 'freshly deployed, minimal real drift'):",
      weekly.iloc[:4].mean())
print("Last 4 weeks avg trigger rate (proxy for 'maximally stale, most real drift'):",
      weekly.iloc[-4:].mean())