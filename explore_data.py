import pandas as pd

# Load the Bank Marketing dataset
df = pd.read_csv(
    "data/bank-additional/bank-additional-full.csv",
    sep=";"
)

# Display basic information
print("DATASET INFORMATION")
print("-------------------")

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["y"].value_counts())