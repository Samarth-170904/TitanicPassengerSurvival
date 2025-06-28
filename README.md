# 🚢 Titanic Passenger Survival Prediction

This project predicts the survival of Titanic passengers using a machine learning pipeline built with scikit-learn. It demonstrates end-to-end data preprocessing, feature transformation, model training, and evaluation using cross-validation — all within a streamlined `Pipeline`.

---

## 📁 Dataset

The dataset used is the classic [Titanic dataset](https://www.kaggle.com/c/titanic/data) from Kaggle. It contains demographic and travel information for passengers aboard the Titanic, including whether they survived.

---

## 📌 Features Used

- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**
- **Age**
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**
- **Embarked**: Port of embarkation

Columns like `PassengerId`, `Name`, `Ticket`, and `Cabin` were dropped as part of preprocessing.

---

## 🧠 ML Workflow

### 🔧 Preprocessing Steps:
- Imputation:
  - Missing `Age`: Replaced with mean.
  - Missing `Embarked`: Replaced with most frequent value.
- One-Hot Encoding: `Sex` and `Embarked` columns.
- Feature Scaling: Applied `MinMaxScaler` to numerical features.

### 🤖 Model:
- **DecisionTreeClassifier**
- Hyperparameter tuning using `GridSearchCV` on:
  - `max_depth`
  - `min_samples_split`
  - `criterion`

---

## 📈 Evaluation

- Accuracy before cross-validation: ~63%
- Accuracy after 5-fold cross-validation: Varies based on hyperparameters.

> Accuracy may improve with further feature engineering, ensemble methods, or model tuning.

---

## 🧪 Getting Started

### 📦 Requirements
```bash
pip install pandas scikit-learn numpy
