# src/ingestion/run_production_ingestion.py
"""Download raw TLC + weather data for the simulated production period."""
from .download_data import load_params, run_ingestion

if __name__ == "__main__":
    params = load_params()
    run_ingestion(
        months=params["data"]["production_months"],
        weather_dest_filename="weather_hourly_production.csv",
    )