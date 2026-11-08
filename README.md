
# Bank Marketing Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a term deposit based on customer information and marketing campaign details.

Five machine learning classification algorithms are trained and evaluated using the same Bank Marketing dataset. An interactive Streamlit application allows users to upload test data, select a trained model, generate predictions, and view evaluation results.

## 2. Dataset Description

Dataset: Bank Marketing (Bank Additional dataset)

Source: UCI Machine Learning Repository

Problem Type: Binary Classification

Target Variable: y

Target Classes:
- yes: Customer subscribes to a term deposit
- no: Customer does not subscribe

The dataset contains customer demographic details, previous marketing interactions, campaign information, and economic indicators.

Example features:
- age
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- duration
- campaign
- pdays
- previous
- poutcome
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

The test dataset contains 8,238 records.

## 3. GitHub Repository

Repository URL: To be added after GitHub upload.

## 4. Classification Models

The following five models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model was evaluated using:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.9000 | 0.8094 | 0.6333 | 0.2662 | 0.3748 | 0.3671 |
| KNN | 0.8972 | 0.7441 | 0.5839 | 0.3039 | 0.3997 | 0.3719 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| SVM | 0.9035 | 0.7060 | 0.6950 | 0.2554 | 0.3735 | 0.3828 |

## 6. Model Performance Observations

### Logistic Regression

Logistic Regression achieved 90.09% accuracy and an AUC of 0.8008. Its precision was relatively high, but its recall of 0.2188 indicates that it missed many actual positive cases.

### Decision Tree

The Decision Tree achieved 90.00% accuracy and the highest AUC score of 0.8094. Its recall was better than Logistic Regression, although it still missed many positive cases.

### K-Nearest Neighbors

KNN achieved 89.72% accuracy with an F1 score of 0.3997. It provided a better balance of precision and recall than Logistic Regression and SVM at the evaluated threshold.

### Naive Bayes

Naive Bayes achieved 80.49% accuracy but had the highest recall of 0.6347 and highest F1 score of 0.4230. It identified more positive customers, although its lower precision indicates more false-positive predictions.

### Support Vector Machine

SVM achieved the highest accuracy of 90.35%, highest precision of 0.6950, and highest MCC of 0.3828. However, its recall remained relatively low.

## 7. Overall Winning Model

SVM is selected as the overall winner based on its highest accuracy and MCC.

However, the best model depends on the business objective. Naive Bayes may be more suitable when identifying as many potential subscribers as possible is the priority, while Decision Tree achieved the highest AUC.

The test data is imbalanced, so accuracy alone should not determine the best classifier.

## 8. Streamlit Application

The interactive Streamlit application provides:

- CSV test dataset upload
- Classification model selection
- Loading of saved trained models
- Predictions on uploaded test data
- Evaluation metrics
- Classification reports
- Confusion matrices
- Model performance comparison
- Accuracy and F1 comparison charts

The application loads previously trained models without retraining them on the uploaded test data.

## 9. Project Files

- app.py
- requirements.txt
- README.md
- test_data.csv
- explore_data.py
- prepare_data.py
- train_logistic.py
- train_decision_tree.py
- train_knn.py
- train_naive_bayes.py
- train_svm.py
- model/

## 10. Running the Application Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## 11. Deployment

Platform: Streamlit Community Cloud

Live Application URL: To be added after deployment.

## 12. Conclusion

All five classification models were successfully implemented and evaluated on the Bank Marketing dataset.

The results demonstrate that different models offer different trade-offs between accuracy, precision, recall, and overall classification performance.

The Streamlit application provides an interactive interface for comparing the trained models and evaluating predictions.
