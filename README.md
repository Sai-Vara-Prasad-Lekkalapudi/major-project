Financial Feature Optimization for Corporate Bankruptcy Prediction
Using Gradient Boosting Models on the Polish Companies Bankruptcy Dataset

Project Overview

This project investigates which financial indicators are most influential in predicting corporate bankruptcy, and how techniques like feature selection, dimensionality reduction, and hyperparameter tuning affect model performance.

I used the Polish Companies Bankruptcy Dataset from the UCI Machine Learning Repository, which contains financial ratios for companies 1–5 years prior to bankruptcy.

My goal was to:

Identify key financial predictors of bankruptcy

Evaluate how boosting models perform across different forecast horizons

Analyze how PCA and preprocessing improve interpretability

Compare model performance between baseline and optimized versions

 Research Question

“Which financial indicators most influence bankruptcy prediction, and how does feature selection affect model performance across different years?"

Dataset

Source: UCI ML Repository
 https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

The dataset is split into five files:

Dataset	Horizon	Rows	Features
1year.arff	Bankruptcy next year	7027	64
2year.arff	2-year horizon	10,500+	64
3year.arff	3-year horizon	10,800+	64
4year.arff	4-year horizon	9,000+	64
5year.arff	5-year horizon	8,000+	64

The class imbalance is severe (≈3–5% bankrupt firms), making this a challenging real-world classification problem.

 Methodology
Data Preprocessing

I applied a fully automated, multi-step pipeline:

Median imputation (missing values reduced to 0)

Outlier removal using IQR-based clipping

Feature scaling using StandardScaler

Dimensionality reduction with PCA (95% variance retained)

Class imbalance correction using SMOTE

Machine Learning Models Used

I evaluated three gradient boosting models:

1. XGBoost

Handles imbalance well

Strong non-linear modeling

Best mid-term predictor (2–4 years)

2. LightGBM

Fast, memory efficient

Best short-term model (1-year dataset)

Best 2-year model after tuning

3. CatBoost

Handles noisy and correlated features

Best long-term model (5-year dataset)

Hyperparameter Tuning

I optimized all models using:

RandomizedSearchCV → broad search

GridSearchCV → fine-tuning

5-fold cross-validation

AUC used as primary metric (due to heavy imbalance)

 Best Results After Tuning
Dataset	Best Model	AUC
1-Year	LightGBM	0.667
2-Year	LightGBM	0.609
3-Year	XGBoost	0.626
4-Year	XGBoost	0.655
5-Year	CatBoost	0.816

 Conclusion:

Short-term predictions - LightGBM

Mid-term predictions - XGBoost

Long-term predictions - CatBoost

No single model dominates across all horizons, proving financial patterns change over time.

Feature Importance Analysis

I extracted top features from each model and built a cross-year feature importance matrix.

Key indicators consistently appearing across years:

Attr15 – profitability/efficiency metric

Attr21 – debt servicing capacity

Attr37 – long-term financial stability

Attr59 – liquidity & leverage indicator

These represent stable, long-term signals of financial distress.

 Visualizations Included

Missing value bar chart

Outlier before/after boxplots

PCA explained variance plot

SMOTE class balance

Model performance comparison plots

Cross-year feature importance heatmap

All plots are generated via Python (Matplotlib + Seaborn).

 Results Summary

Boosting models outperform classical ML methods on this dataset

Preprocessing (SMOTE + PCA) significantly improves model stability

Feature importance varies by horizon, but key indicators remain consistent

CatBoost and XGBoost excel at long-horizon forecasting

LightGBM gives the best short-term interpretability and accuracy
