import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load the dataset
df = pd.read_csv(
    "data/bank-additional/bank-additional-full.csv",
    sep=";"
)

# 2. Separate input and output
X = df.drop(columns=["y", "duration"])
y = (df["y"] == "yes").astype(int)

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Identify feature types
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 5. Prepare preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_columns),
    ("numerical", StandardScaler(), numerical_columns)
])

# 6. Build Logistic Regression pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train the model
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# 8. Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 9. Calculate evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": roc_auc_score(y_test, probabilities),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1": f1_score(y_test, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions)
}

print("\nLOGISTIC REGRESSION RESULTS")
print("---------------------------")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 10. Save the trained pipeline
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/logistic_regression.joblib")

print("\nModel saved successfully!")