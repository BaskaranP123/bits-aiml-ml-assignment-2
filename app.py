

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing - ML Classification")
st.write("BITS Pilani | Machine Learning Assignment 2")

st.markdown("""
This application evaluates five machine learning
classification models using the Bank Marketing dataset.
""")

st.subheader("Step 1: Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset dimensions:", df.shape)

    st.dataframe(df.head())

    st.subheader("Step 2: Select Classification Model")

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }

    selected_model = st.selectbox(
        "Choose a trained ML model",
        list(model_files.keys())
    )

    model_path = Path("model") / model_files[selected_model]

    if model_path.exists():
        model = joblib.load(model_path)
        st.success(f"{selected_model} loaded successfully!")
    else:
        st.error(f"Model file not found: {model_path}")
        st.stop()

    
    # STEP 3: Model Prediction and Evaluation
    st.subheader("Step 3: Model Prediction and Evaluation")

    if st.button("Run Prediction"):
        try:
            if "y" not in df.columns:
                st.error("The uploaded CSV must contain the 'y' column.")
                st.stop()

            # Separate input features and actual target
            X_test = df.drop(columns=["y"])
            y_test = df["y"].replace({"no": 0, "yes": 1}).astype(int)

            # Generate predictions
            predictions = model.predict(X_test)
            predictions = pd.Series(predictions).replace(
                {"no": 0, "yes": 1}
            ).astype(int).to_numpy()

            st.success("Predictions generated successfully!")

            # Display prediction results
            results = df.copy()
            results["Predicted"] = predictions

            st.write("Prediction Results")
            st.dataframe(results.head(20))

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, zero_division=0
            )
            mcc = matthews_corrcoef(y_test, predictions)

            st.subheader("Model Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")

            col4, col5 = st.columns(2)

            col4.metric("F1 Score", f"{f1:.4f}")
            col5.metric("MCC", f"{mcc:.4f}")

            # Calculate AUC using prediction probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_test)[:, 1]
                if y_test.nunique() == 2:
                    auc = roc_auc_score(y_test, probabilities)
                    st.metric("ROC-AUC", f"{auc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )

            fig, ax = plt.subplots()
            image = ax.imshow(cm, cmap="Blues")
            fig.colorbar(image, ax=ax)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No", "Yes"])
            ax.set_yticklabels(["No", "Yes"])

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center"
                    )

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")
            report = classification_report(
                y_test, predictions, zero_division=0
            )
            st.text(report)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.subheader("Step 4: Model Performance Comparison")



comparison_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ],
    "Accuracy": [0.9009, 0.9000, 0.8972, 0.8049, 0.9035],
    "AUC": [0.8008, 0.8094, 0.7441, 0.7755, 0.7060],
    "Precision": [0.6905, 0.6333, 0.5839, 0.3172, 0.6950],
    "Recall": [0.2188, 0.2662, 0.3039, 0.6347, 0.2554],
    "F1 Score": [0.3322, 0.3748, 0.3997, 0.4230, 0.3735],
    "MCC": [0.3516, 0.3671, 0.3719, 0.3490, 0.3828]
}


comparison_df = pd.DataFrame(comparison_data)


comparison_df = pd.DataFrame(comparison_data)

st.dataframe(comparison_df, use_container_width=True)

st.subheader("Model Accuracy Comparison")
st.bar_chart(comparison_df.set_index("Model")["Accuracy"])

st.subheader("Model F1 Score Comparison")
st.bar_chart(comparison_df.set_index("Model")["F1 Score"])




