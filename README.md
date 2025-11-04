# Financial Feature Optimization for Corporate Bankruptcy Prediction Using Gradient Boosting
Overview
This project analyzes and optimizes feature selection for predicting corporate bankruptcy using the Polish Companies Bankruptcy dataset and advanced machine learning (gradient boosting) techniques. It focuses on identifying the most important financial indicators for early bankruptcy prediction, evaluating the effect of feature selection and dimensionality reduction strategies, and addressing common challenges in financial data like class imbalance, missing data, and outlier effects.

Dataset
Source: Polish Companies Bankruptcy Data (UCI Machine Learning Repository)

Files: 5 forecast horizons (1-year to 5-year to bankruptcy)

Each file: 7027-10503 samples, 64 numerical attributes (Attr1-Attr64), 1 binary target (class: 0 = not bankrupt, 1 = bankrupt)

Project Goals
Research Question: Which financial indicators are most influential in predicting corporate bankruptcy, and how does feature selection affect model performance?

Objectives:

Identify top financial indicators affecting bankruptcy risk.

Compare model accuracy and interpretability with various feature selection and dimensionality reduction techniques.

Address real-world dataset challenges (missing values, outlier effects, class imbalance).

Project Steps
1. Data Loading & Initial Exploration
Load all 5 datasets (1-5 year prediction files).

Consistent schema across files: Attr1–Attr64 + class label.

Statistics and data info reviewed; observed significant class imbalance and missing values.

2. Data Preprocessing
Missing Value Imputation: Used median imputation for all numeric features (to preserve key financial indicators).

Class Label Cleaning: Converted binary labels to numeric (0/1).

3. Exploratory Data Analysis (EDA)
Class Distribution: Visualizations show heavy imbalance (fewer bankrupt companies).

Missing Data Visualization: Heatmaps and summary tables.

Feature Distribution: Boxplots for important attributes segmented by class.

Correlation Analysis: Heatmaps for top influential features.

4. Data Preparation
Class Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority (bankrupt) class for balanced training.

Outlier Treatment: Applied IQR-based clipping to limit extreme financial attribute outliers.

Feature Scaling: StandardScaler for model input standardization.

5. Feature Selection & Modeling
(Intended workflow—customize based on your code output)

Test Feature Selection: Recursive Feature Elimination (RFE), PCA, SHAP.

Train Gradient Boosting and compare with baseline models (Logistic Regression, SVM, KNN).

Evaluate impact of feature selection on accuracy, AUC, and interpretability.

Results
Imputation resolved all missing values.

SMOTE produced balanced training sets for robust learning.

IQR Clipping and Scaling normalized feature space.
