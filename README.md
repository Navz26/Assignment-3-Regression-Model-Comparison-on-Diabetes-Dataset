Assignment 3 – Regression Model Comparison on Diabetes Dataset


Name: Navpreet Kaur Badhwal
Student ID: 300212193


1)  Project Overview
This assignment explores and compares linear and nonlinear regression models using the Diabetes dataset. The goal is to evaluate how different modeling techniques perform when predicting a continuous target variable and to understand the strengths and weaknesses of each approach.
Workflow & Code Explanation
1. Import Libraries
Required Python libraries are imported for data handling, visualization, preprocessing, and modeling.
2. Load Dataset
The Diabetes dataset is loaded into a pandas DataFrame for analysis.
3. Initial Data Exploration
Displayed first few rows
Reviewed summary statistics
Checked for missing values
4. Exploratory Data Analysis (EDA)
Plotted histograms to observe feature distributions
Generated a correlation heatmap to understand relationships between variables
5. Stratified Sampling
The dataset is split into training and testing sets using stratified sampling to maintain the distribution of the target variable.
6. Feature and Target Separation
Features (X) and target variable (y) are separated for modeling
7. Feature Scaling
All features are standardized to ensure fair comparison, especially for models sensitive to scale.


2)  Models Implemented
Linear Regression
Ridge Regression (L2 Regularization)
Polynomial Regression (Degree 3 and Degree 5)
Support Vector Regression (SVR)
Random Forest Regressor
Each model is trained and tested using the same dataset split for consistency.


3)  Evaluation Metric
Root Mean Squared Error (RMSE)
RMSE is used to evaluate and compare model performance.
RMSE Results
Model	RMSE
Linear Regression	54.54
Ridge Regression	54.27
Polynomial Regression (Degree 3)	210.61
Polynomial Regression (Degree 5)	295.84
Support Vector Regression (SVR)	51.89
Random Forest Regressor	53.23


4)  Model Performance Discussion
Linear Models
Linear and Ridge regression show moderate performance
Slight underfitting due to limited ability to model complex relationships
Polynomial Regression
High-degree polynomial models show severe overfitting
Performance degrades significantly as degree increases
Nonlinear Models
SVR delivers the best overall performance and generalizes well
Random Forest balances bias and variance effectively and handles nonlinearity


5)  Linear vs Nonlinear Models
Linear models are simple and interpretable but struggle with complex patterns
Nonlinear models capture complex relationships more effectively
Polynomial regression overfits easily at high degrees
SVR and Random Forest provide strong generalization on unseen data


6)  Approaches to Improve Performance
Use GridSearchCV to tune model hyperparameters
Perform feature selection to remove noisy or irrelevant features
Apply cross-validation for more reliable performance estimates
Limit polynomial degree or apply regularization
Use ensemble or stacking techniques to combine multiple models


7)  Tools & Technologies
Python
pandas
NumPy
scikit-learn
matplotlib / seaborn


8)  Course-
COMP 381- Machine learning 
Assignment 3
