import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load dataset
df = pd.read_csv(
    "data/bank-additional/bank-additional-full.csv",
    sep=";"
)

# Prepare features and target
X = df.drop(columns=["y", "duration"])
y = (df["y"] == "yes").astype(int)

# Same split as Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify feature types
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_columns),
    ("numerical",
     StandardScaler(),
     numerical_columns)
])

# Build Decision Tree pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        random_state=42
    ))
])

print("Training Decision Tree...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": roc_auc_score(y_test, probabilities),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, predictions, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_test, predictions)
}

print("\nDECISION TREE RESULTS")
print("---------------------")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Save trained model
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/decision_tree.joblib")

print("\nDecision Tree model saved successfully!")