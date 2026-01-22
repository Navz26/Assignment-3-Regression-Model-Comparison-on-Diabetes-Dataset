# SEE 895 Assignment #3: Regression on Diabetes Dataset

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#  Loading Dataset
diabetes = datasets.load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

# Quick look at data
print(df.head())
print(df.describe())
print(df.isnull().sum())  


#  Exploratory Data Analysis

# Histograms of features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("histograms.png")  
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig("correlation_heatmap.png")  
plt.show()


#  Stratified Random Sampling

# Bin target variable into 5 categories for stratification
df['target_cat'] = pd.cut(df['target'], bins=5, labels=False)

# Use last two digits of student number as seed
random_seed = 93

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
for train_index, test_index in split.split(df, df['target_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# Drop the categorical column after split
for dataset in (strat_train_set, strat_test_set):
    dataset.drop("target_cat", axis=1, inplace=True)

# Separate features and target
X_train = strat_train_set.drop("target", axis=1)
y_train = strat_train_set["target"]
X_test = strat_test_set.drop("target", axis=1)
y_test = strat_test_set["target"]


#  Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  Linear Regression

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print("Linear Regression RMSE:", rmse_lin)


#  Ridge Regression

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression RMSE:", rmse_ridge)


#  Polynomial Regression 
for degree in [3, 5]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)
    
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    print(f"Polynomial Regression (degree {degree}) RMSE:", rmse_poly)


#  Nonlinear Regression (SVR)
svr_reg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_reg.fit(X_train_scaled, y_train)
y_pred_svr = svr_reg.predict(X_test_scaled)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
print("SVR RMSE:", rmse_svr)


#  Ensemble Method (Random Forest Regressor)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=random_seed)
rf_reg.fit(X_train, y_train)  # RF handles unscaled data
y_pred_rf = rf_reg.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", rmse_rf)
