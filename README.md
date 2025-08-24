House Price Prediction - Kaggle Competition
Overview
This project is a solution for the Kaggle House Prices - Advanced Regression Techniques competition. The goal is to predict house sale prices (SalePrice) based on various features like lot size, neighborhood, and overall quality. The project includes exploratory data analysis (EDA), preprocessing, feature engineering, model training, and visualization, achieving a cross-validation RMSE of ~0.14 (log scale) using XGBoost.
Dataset
The dataset, provided by Kaggle, includes:

train.csv: Training data with 1460 rows and 81 columns (80 features + SalePrice).
test.csv: Test data with 1459 rows and 80 features.
data_description.txt: Description of each feature.
sample_submission.csv: Benchmark submission.

Key features include:

SalePrice: Target variable (house sale price in dollars).
OverallQual: Overall material and finish quality.
GrLivArea: Above-ground living area in square feet.
TotalBsmtSF: Total basement square feet.
Neighborhood: Physical location within Ames city limits.

Requirements
To run the code, install the required Python packages:
pip install pandas numpy scikit-learn xgboost seaborn matplotlib

Project Structure

train.csv: Training dataset.
test.csv: Test dataset.
submission.csv: Output file with predictions for Kaggle submission.
house_price_prediction.ipynb: Jupyter Notebook with the full code (EDA, preprocessing, modeling, visualizations).
README.md: This file.

Methodology
1. Exploratory Data Analysis (EDA)

Missing Values: Identified and handled missing values in columns like Electrical (1 in train), MSZoning (4 in test), and others.
SalePrice Distribution: Visualized the distribution of SalePrice, which is right-skewed (log-transformed for modeling).
Correlations: Analyzed correlations with SalePrice, identifying key features like OverallQual and GrLivArea.
Categorical Features: Explored relationships (e.g., Neighborhood vs. SalePrice) using boxplots.

2. Preprocessing

Missing Value Imputation:
Categorical: Imputed with mode (e.g., Electrical with "SBrkr", MSZoning with "RL").
Numerical: Imputed with median (e.g., BsmtFullBath, BsmtHalfBath).
Features like PoolQC and Alley imputed with "None" for absent features.


Feature Engineering:
Created TotalSF (sum of TotalBsmtSF, 1stFlrSF, 2ndFlrSF).
Added HouseAge (YrSold - YearBuilt) and RemodAge (YrSold - YearRemodAdd).
Log-transformed skewed features (LotArea, GrLivArea, TotalSF) and SalePrice.


Encoding: Applied one-hot encoding to categorical features (e.g., Neighborhood, MSZoning).
Scaling: Standardized features using StandardScaler.

3. Model Training

Models Evaluated:
Linear Regression: RMSE ~23 billion (likely due to multicollinearity/outliers).
Random Forest: RMSE ~0.1457.
XGBoost: RMSE ~0.1402 (best model).


Evaluation: Used 5-fold cross-validation with RMSE on log-transformed SalePrice.
Selected Model: XGBoost, due to its superior performance.

4. Visualizations

Feature Importance: Plotted top 10 features (e.g., TotalSF, OverallQual) for XGBoost.
Predicted vs Actual: Scatter plot of predicted vs actual SalePrice on the training set.

5. Submission

Generated predictions using XGBoost on the test set.
Reversed log transformation (np.expm1) for submission.
Created submission.csv with columns Id and SalePrice.

How to Run

Clone the Repository:git clone <repository-url>
cd <repository-folder>


Install Dependencies:pip install -r requirements.txt

(Create requirements.txt with the listed packages if needed.)
Run the Notebook:
Open house_price_prediction.ipynb in Jupyter Notebook or JupyterLab.
Ensure train.csv and test.csv are in the same directory.
Run all cells to perform EDA, preprocessing, modeling, and generate submission.csv.


Submit to Kaggle:
Upload submission.csv to the Kaggle competition.
Expected leaderboard RMSE: ~0.13–0.15.



Results

Cross-Validation RMSE:
Linear Regression: ~23 billion (needs debugging).
Random Forest: ~0.1457.
XGBoost: ~0.1402.


Key Features: TotalSF, OverallQual, GrLivArea, and Neighborhood are top predictors.
Submission: Predictions saved in submission.csv for Kaggle.

Next Steps

Improve Linear Regression:
Use Ridge or Lasso regression to handle multicollinearity.
Remove outliers (e.g., SalePrice > 500,000).


Feature Engineering:
Add interaction terms (e.g., OverallQual * TotalSF).
Bin HouseAge into categories.


Model Tuning:
Optimize XGBoost parameters using GridSearchCV.
Try LightGBM or ensemble methods.


Outlier Handling:
Remove extreme SalePrice values to improve robustness.



License
This project is licensed under the MIT License.
Acknowledgments

Kaggle for providing the dataset and competition.
Inspired by common practices in the Kaggle community for house price prediction.
