
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# STEP 1: Load dataset
df = pd.read_csv(
    "data/bank-additional/bank-additional-full.csv",
    sep=";"
)

# STEP 2: Separate features and target
X = df.drop(columns=["y", "duration"])
y = (df["y"] == "yes").astype(int)

# STEP 3: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 4: Identify categorical and numerical features
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# STEP 5: Preprocessing
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_columns
    ),
    (
        "numerical",
        StandardScaler(),
        numerical_columns
    )
])

# STEP 6: Create Naive Bayes pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", GaussianNB())
])

# STEP 7: Train model
print("Training Naive Bayes Classifier...")
model.fit(X_train, y_train)

# STEP 8: Predict test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# STEP 9: Evaluate model
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

print("\nNAIVE BAYES RESULTS")
print("----------------------")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# STEP 10: Save trained model
Path("model").mkdir(exist_ok=True)

joblib.dump(model, "model/naive_bayes.joblib")

print("\nNaive Bayes model saved successfully!")
