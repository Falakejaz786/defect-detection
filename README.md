# Defect Detection using XGBoost and SMOTE

## Overview

This project focuses on predicting manufacturing defects using Machine Learning.
The goal is to identify defective coils while minimizing false negatives, which is critical in industrial quality control systems.

The model is built using:

* XGBoost Classifier
* SMOTE for handling class imbalance
* Median Imputation for missing values
* Threshold tuning for maximizing recall

---

## Problem Statement

Manufacturing industries generate large amounts of process data.
The objective of this project is to predict whether a coil contains an Alpha defect (`Y = 1`) or not (`Y = 0`) using process parameters.

### Key Requirements

* Minimize false negatives
* Achieve very high recall
* Maintain good precision

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

---

## Project Workflow

### 1. Data Loading

* Loaded training and testing datasets using Pandas.

### 2. Data Preprocessing

* Removed unnecessary columns like `CoilID`
* Separated features and target variable

### 3. Train-Test Split

* Split the dataset into training and validation sets using stratified sampling.

### 4. Missing Value Handling

* Used `SimpleImputer(strategy="median")` to replace missing values.

### 5. Handling Class Imbalance

* Applied SMOTE to balance minority defect samples.

### 6. Model Training

* Trained an XGBoost classifier with tuned hyperparameters.

### 7. Threshold Optimization

* Tested multiple probability thresholds to maximize recall and reduce false negatives.

### 8. Final Prediction

* Generated predictions on test data and prepared submission file.

---

## Model Features

* XGBoost for powerful classification
* SMOTE for synthetic minority oversampling
* Threshold tuning for recall optimization
* Imputation for missing values
* Class imbalance handling using `scale_pos_weight`

---

## Evaluation Metrics

The following metrics were used:

* Recall Score
* Precision Score
* Confusion Matrix
* Classification Report

Special focus was given to:

[
Recall = \frac{TP}{TP + FN}
]

to ensure defect samples are not missed.

---

## Project Structure

```bash
defect-detection-project/
│
├── defect_detection.py
├── train.csv
├── test.csv
├── requirements.txt
└── README.md
```

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
```

---

## Running the Project

```bash
python defect_detection.py
```

---

## Output

The model generates:

* Defect predictions
* Optimized threshold-based classifications
* Submission dataframe containing:

  * CoilID
  * Predicted label (`Y`)

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Feature engineering
* Cross-validation
* Ensemble learning
* Deployment using Flask or Streamlit

---
