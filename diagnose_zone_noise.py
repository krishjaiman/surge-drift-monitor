import json
import pandas as pd
from collections import Counter

with open("data/monitoring/alerts_log.jsonl") as f:
    records = [json.loads(l) for l in f.readlines()]

bias_counter = Counter()
calib_counter = Counter()
for r in records:
    bias_counter.update(r["breach_details"]["zone_bias_breaches"])
    calib_counter.update(r["breach_details"]["zone_calibration_breaches"])

print("=== Top 15 zones by bias-alert frequency ===")
for zone, count in bias_counter.most_common(15):
    print(f"zone {zone}: {count} hours flagged")
print()
print("=== Top 15 zones by calibration-alert frequency ===")
for zone, count in calib_counter.most_common(15):
    print(f"zone {zone}: {count} hours flagged")
print()

# Average trip volume for the most frequently-flagged zones, from demand_history
import sqlite3
conn = sqlite3.connect("data/serving/demand_history.db")
top_bias_zones = [z for z, _ in bias_counter.most_common(10)]
placeholders = ",".join("?" * len(top_bias_zones))
df = pd.read_sql_query(
    f"SELECT zone_id, AVG(trip_count) as avg_trips FROM demand_history "
    f"WHERE zone_id IN ({placeholders}) AND is_actual = 1 GROUP BY zone_id",
    conn, params=top_bias_zones,
)
conn.close()
print("=== Avg hourly trip volume for most-flagged bias zones ===")
print(df.sort_values("avg_trips"))

print()
print("Overall: how many distinct zones ever appeared in a bias breach:", len(bias_counter))
print("Overall: how many distinct zones ever appeared in a calibration breach:", len(calib_counter))