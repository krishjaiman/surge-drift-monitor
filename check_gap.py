# check_gap.py
import pandas as pd
import sqlite3
from pathlib import Path

# 1. Does weather_hourly.parquet exist, and what does it cover?
weather_path = Path("data/processed/weather_hourly.parquet")
if weather_path.exists():
    w = pd.read_parquet(weather_path)
    print("WEATHER CACHE:")
    print("  rows:", len(w))
    print("  min hour_ts:", w["hour_ts"].min())
    print("  max hour_ts:", w["hour_ts"].max())
    print("  covers 2024-07-07?:", pd.Timestamp("2024-07-07 19:00:00") in set(w["hour_ts"]))
else:
    print("WEATHER CACHE FILE DOES NOT EXIST:", weather_path)

# 2. What's actually in demand_history around the gap?
conn = sqlite3.connect("data/serving/demand_history.db")
print("\nDEMAND HISTORY:")
print("  total rows:", conn.execute("SELECT COUNT(*) FROM demand_history").fetchone())
print("  min ts:", conn.execute("SELECT MIN(ts) FROM demand_history").fetchone())
print("  max ts:", conn.execute("SELECT MAX(ts) FROM demand_history").fetchone())
print("  rows for zone 132 around 2024-07-07:")
rows = conn.execute(
    "SELECT ts, trip_count FROM demand_history WHERE zone_id=132 AND ts BETWEEN '2024-06-25' AND '2024-07-08' ORDER BY ts"
).fetchall()
for r in rows[:20]:
    print("   ", r)
print("  count in that window:", len(rows))
conn.close()