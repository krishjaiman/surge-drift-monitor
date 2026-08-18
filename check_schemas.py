import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
print("COLUMNS:")
print(df.columns.tolist())
print("\nDTYPES:")
print(df.dtypes)
print("\nSAMPLE ROWS:")
print(df.head(3).to_string())
print("\nSHAPE:", df.shape)