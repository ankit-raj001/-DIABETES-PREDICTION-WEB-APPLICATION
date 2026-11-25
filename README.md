# End-to-End Diabetes Prediction Web Application

An end-to-end clinical decision-support web application that predicts the risk of Diabetes Mellitus from patient biometric indicators. Built with **Scikit-Learn** and deployed interactively via **Streamlit**, the project covers clinical data preprocessing, supervised binary classification, model serialization, and real-time browser-based inference.

## Overview
Early detection of Diabetes Mellitus helps prevent acute metabolic complications as well as long-term cardiovascular and renal issues. This application uses eight biometric features to provide fast, non-invasive diabetes risk screening through an interactive web interface.

## Key Features & Engineering Highlights
- **Biological Zero-Value Imputation:** Detected physiologically invalid zero values across five clinical features (`Insulin` 48.7%, `SkinThickness` 29.6%, `BloodPressure` 4.6%, `BMI` 1.4%, and `Glucose` 0.7%) and replaced them with `np.nan` followed by median imputation—preserving all 768 patient records without outlier distortion.
- **Model Evaluation & Selection:** Compared supervised classifiers (Logistic Regression, Random Forest, Support Vector Machine, K-Nearest Neighbors, and Decision Tree) on an 80:20 train-test split (`614` train / `154` test samples, `random_state=42`), selecting **Logistic Regression** for strong generalization (`77.3%` test accuracy, `83.2%` ROC-AUC) and interpretability.
- **Model Persistence & Web App:** Exported the trained model using Python `pickle` (`diabetes_prediction.pkl`) and built a reactive **Streamlit** interface (`app.py`) with bounded biometric inputs and diagnostic feedback banners.

## Dataset Attributes
Trained on the **Pima Indians Diabetes Database** (NIDDK / UCI / Kaggle — `768` records, `8` diagnostic predictors, binary `Outcome` target):

| Attribute | Type | Clinical Role & Unit |
| :--- | :--- | :--- |
| `Pregnancies` | Integer | Number of lifetime pregnancies (count) |
| `Glucose` | Integer / Float | 2-hour oral glucose tolerance test plasma concentration (`mg/dL`) |
| `BloodPressure` | Integer / Float | Diastolic blood pressure (`mm Hg`) |
| `SkinThickness` | Integer / Float | Triceps skin fold thickness (`mm`) |
| `Insulin` | Integer / Float | 2-hour post-load serum insulin (`mu U/ml`) |
| `BMI` | Float | Body Mass Index (`kg/m²`) |
| `DiabetesPedigreeFunction` | Float | Genealogical diabetes risk score |
| `Age` | Integer | Patient age (`years`) |

## Model Benchmark Summary (154 Holdout Test Samples)

| Algorithm | Train Accuracy | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **78.5%** | **77.3%** | **72.1%** | **57.4%** | **63.9%** |
| Random Forest (100 Trees) | 98.5% | 76.6% | 69.0% | 55.5% | 61.5% |
| Support Vector Machine (RBF) | 82.4% | 74.0% | 66.7% | 51.8% | 58.3% |
| K-Nearest Neighbors (`k=5`) | 81.2% | 72.7% | 62.5% | 55.5% | 58.8% |
| Decision Tree (CART) | 100.0% | 68.8% | 55.1% | 50.0% | 52.4% |

## Tech Stack
- **Python 3.12**
- **Pandas & NumPy** (data cleaning, imputation, array manipulation)
- **Scikit-Learn & Pickle** (model training, evaluation, serialization)
- **Streamlit** (interactive web UI and inference)

## Repository Structure
```text
├── diabetes.csv                 # Pima Indians Diabetes Dataset (768 rows, 9 columns)
├── diabetes_prediction.ipynb    # Exploratory analysis, preprocessing, training & evaluation
├── diabetes_prediction.pkl    # Serialized Logistic Regression model
├── app.py                       # Streamlit web application
└── requirements.txt             # Project dependencies




