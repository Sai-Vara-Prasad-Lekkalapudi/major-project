Financial Feature Optimization for Corporate Bankruptcy Prediction using Gradient Boosting Models
Project Overview

This project investigates the use of machine learning techniques to predict corporate bankruptcy using financial ratio data. The objective is to identify the most influential financial indicators associated with bankruptcy risk and evaluate the predictive performance of gradient boosting models across multiple forecasting horizons (1–5 years before bankruptcy).

An end-to-end machine learning pipeline was implemented including exploratory data analysis, data preprocessing, model training, hyperparameter optimisation, evaluation, and model explainability using SHAP.

Research Question

Which financial indicators are most influential in predicting corporate bankruptcy, and how does model optimisation affect predictive performance across different forecasting horizons?

Dataset

The project uses the Polish Companies Bankruptcy Dataset from the UCI Machine Learning Repository.

Dataset characteristics:

64 financial ratio features

1 binary target variable (bankrupt or non-bankrupt)

Five datasets representing 1–5 years before bankruptcy

Strong class imbalance (bankrupt firms ≈ 3–5%)

Dataset link:
https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

The financial ratios capture several dimensions of corporate financial health:

profitability

liquidity

leverage

solvency

operational efficiency

Examples of important financial ratios include:

Attr1 – net profit / total assets

Attr7 – operating profit / total assets

Attr27 – working capital / total assets

These ratios are widely used indicators in bankruptcy prediction research.

Project Workflow
1. Exploratory Data Analysis (EDA)

Initial data exploration was performed to understand dataset structure and quality.

Key steps included:

dataset inspection

missing value analysis

class imbalance analysis

correlation analysis of financial ratios

EDA helped identify multicollinearity between financial variables and confirmed the need for dimensionality reduction.

2. Data Preprocessing

Several preprocessing steps were applied before model training:

Median imputation to handle missing values

IQR-based clipping to treat outliers

StandardScaler for feature scaling

SMOTE to address class imbalance

PCA to reduce dimensionality and remove multicollinearity

PCA retained 95% of cumulative explained variance, resulting in approximately 24–25 principal components across forecasting horizons.

Machine Learning Models

Three gradient boosting algorithms were evaluated:

XGBoost

LightGBM

CatBoost

Each model was trained using:

Baseline configuration

Hyperparameter tuned configuration

Models were evaluated separately for each forecasting horizon (1–5 years before bankruptcy).

Model Evaluation

Model performance was evaluated using multiple classification metrics:

Accuracy

Precision

Recall

F1-Score

AUC (primary metric)

Additional evaluation techniques included:

confusion matrices

ROC curves

AUC comparison plots

AUC was prioritised because it is robust to class imbalance and measures the model’s ability to distinguish between bankrupt and non-bankrupt firms.

Key Results

The models achieved strong predictive performance across all forecasting horizons.

Main findings:

AUC values consistently exceeded 0.97

CatBoost achieved the highest AUC for the 1-year prediction horizon

XGBoost showed the most stable performance across longer horizons

Hyperparameter tuning produced competitive results, although baseline models already performed strongly

Tuned models achieved average AUC values around 0.985–0.986 across forecasting horizons

These results indicate that gradient boosting models are highly effective for bankruptcy prediction tasks.

Model Explainability (SHAP)

SHAP (SHapley Additive Explanations) was used to interpret the model predictions.

The analysis revealed that a small number of principal components contributed most strongly to bankruptcy predictions, particularly:

PC1

PC3

PC7

PC10

These components were strongly influenced by profitability and liquidity ratios such as Attr1, Attr7, and Attr27, which are key indicators of financial distress.

Repository Structure
project/
│
├── notebook/
│   Sai_Bankruptcy_Final_MSc_Notebook_Updated.ipynb
│
├── report/
│   Financial_Bankruptcy_Prediction_Report.pdf
│
├── data/
│   1year.arff
│   2year.arff
│   3year.arff
│   4year.arff
│   5year.arff
│
└── README.md
Technologies Used

The project was implemented using:

Python

Pandas

NumPy

Scikit-learn

XGBoost

LightGBM

CatBoost

SHAP

Matplotlib

Seaborn

Reproducibility

To reproduce the project results:

Clone the repository

git clone https://github.com/Sai-Vara-Prasad-Lekkalapudi/major-project

Install required libraries

pip install pandas numpy scikit-learn xgboost lightgbm catboost shap seaborn matplotlib

Open the Jupyter notebook

jupyter notebook

Run the notebook cells sequentially to reproduce the analysis, models, and visualisations.

Author

Sai Vara Prasad Lekkalapudi
MSc Data Science
University of Hertfordshire

Supervisor: Philip Lucas

Acknowledgement

This project uses the Polish Companies Bankruptcy Dataset from the UCI Machine Learning Repository for academic research purposes.
