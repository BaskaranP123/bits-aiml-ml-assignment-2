import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Read the dataset
df = pd.read_csv(
    "data/bank-additional/bank-additional-full.csv",
    sep=";"
)

# Step 2: Separate input features and target
X = df.drop(columns=["y"])
y = df["y"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Save test data with the target column
test_data = X_test.copy()
test_data["y"] = y_test

test_data.to_csv("test_data.csv", index=False)

# Step 5: Display results
print("DATA PREPARATION COMPLETED")
print("--------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Input features:", X.shape[1])
print("Test CSV saved: test_data.csv")