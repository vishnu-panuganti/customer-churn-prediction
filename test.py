import pandas as pd

# Load the dataset
df = pd.read_csv("data/tele.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns.tolist())