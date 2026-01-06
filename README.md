Financial Feature Optimisation for Corporate Bankruptcy Prediction

Using Gradient Boosting Models and Explainable AI

📌 Project Overview

This project develops an end-to-end machine learning pipeline to predict corporate bankruptcy using financial ratio data. The study focuses on identifying the most influential financial indicators and evaluating how feature optimisation and explainability affect model performance across multiple forecasting horizons.

I applied modern gradient boosting models (XGBoost, LightGBM, and CatBoost) and combined them with robust preprocessing, hyperparameter optimisation, and SHAP-based explainability to produce accurate, stable, and interpretable bankruptcy prediction models.

🎯 Research Question

Which financial indicators are most influential in predicting corporate bankruptcy, and how does feature optimisation affect the performance and interpretability of machine learning models across different forecasting horizons?

📂 Dataset

Source: UCI Machine Learning Repository

Dataset: Polish Companies Bankruptcy Dataset

Scope:

5 datasets representing 1–5 years before bankruptcy

64 numerical financial ratios

Binary target variable (class: bankrupt / non-bankrupt)

Characteristics:

Strong class imbalance

No personally identifiable information (fully anonymised)

⚙️ Methodology
1. Exploratory Data Analysis (EDA)

Missing value analysis

Class imbalance analysis

Correlation analysis of financial ratios

2. Data Pre-Processing

Median imputation for missing values

Outlier treatment using IQR clipping

Feature scaling with StandardScaler

Class imbalance handling using SMOTE

Dimensionality reduction using PCA (95% variance)

3. Model Development

Baseline Models

XGBoost

LightGBM

CatBoost

Hyperparameter Optimisation

RandomisedSearchCV

AUC as the primary optimisation metric

4. Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

AUC (primary metric)

5. Explainability

SHAP summary (global importance)

SHAP beeswarm (direction + magnitude)

SHAP waterfall (local explanation)

🏆 Key Results
Best Tuned Model (Overall)

Model: LightGBM_Tuned

Mean AUC (1–5 years): 0.984

Best balance of:

Predictive performance

Stability across time horizons

Computational efficiency

Key Observations

Bankruptcy risk can be predicted several years in advance

Gradient boosting models outperform traditional baselines

A small number of components (e.g., PC1, PC3, PC10) dominate predictions

SHAP explanations make complex models transparent and usable in practice

📊 Visualisations Included

AUC comparison (baseline vs tuned)

Heatmaps and trend plots across forecasting horizons

Confusion matrices and ROC curves

SHAP feature importance, beeswarm, and waterfall plots

🧠 Practical Applications

Credit risk assessment

Investment screening

Regulatory early-warning systems

Financial decision support tools

🔮 Future Work

Incorporate macroeconomic indicators

Improve raw feature interpretability without PCA

Evaluate generalisability on datasets from other countries

Explore cost-sensitive learning approaches
