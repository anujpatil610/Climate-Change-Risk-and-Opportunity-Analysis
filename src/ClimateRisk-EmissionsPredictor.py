#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 4 19:22:5 2023

Climate Change Risk and Opportunity Analysis

Authors:Anuj Patil

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Load the new encoded dataset
encoded_data = pd.read_csv('cleaned_data_label_encoded.csv')

# Separate the target variable from the features
X = encoded_data.drop(columns='Total_Emissions')
y = encoded_data['Total_Emissions']

# Lasso Regression for Feature Selection
lasso = Lasso(alpha=0.1) 
lasso.fit(X, y)
lasso_feature_selection = SelectFromModel(lasso, prefit=True)
selected_features = X.columns[lasso_feature_selection.get_support()]

# Keep only the selected features in the dataset
X = X[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store the X_test indices before scaling
X_test_indices = X_test.index

# Perform normalization on the selected features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Define the parameter grid for hyperparameter tuning
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50]
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the randomized search model
 #randomized_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=3, random_state=42, n_jobs=-1, verbose=2)

# Fit the randomized search to the data
#randomized_search.fit(X_train, y_train)

# Check the best parameters
#print(randomized_search.best_params_)

# Create a new Random Forest model with the optimal parameters
optimal_rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)

# Train the new model
optimal_rf.fit(X_train, y_train)

# Make predictions
optimal_y_pred_train = optimal_rf.predict(X_train)
optimal_y_pred_test = optimal_rf.predict(X_test)

# Calculate MAE and R-squared for training data
mae_train = mean_absolute_error(y_train, optimal_y_pred_train)
r2_train = r2_score(y_train, optimal_y_pred_train)

# Calculate MAE and R-squared for testing data
mae_test = mean_absolute_error(y_test, optimal_y_pred_test)
r2_test = r2_score(y_test, optimal_y_pred_test)

print(f'Training MAE: {mae_train}')
print(f'Testing MAE: {mae_test}')
print(f'Training R-squared: {r2_train}')
print(f'Testing R-squared: {r2_test}')

# Ensure all data is in DataFrame format with reset indices
df_X_train = pd.DataFrame(X_train).reset_index(drop=True)
df_y_train = pd.DataFrame(y_train, columns=['Total_Emissions']).reset_index(drop=True)
df_pred_train = pd.DataFrame(optimal_y_pred_train, columns=['Predicted_Emissions']).reset_index(drop=True)

# Concatenate all DataFrames along the column axis
df_train = pd.concat([df_X_train, df_y_train, df_pred_train], axis=1)

# Calculate residuals
df_train['Residuals'] = df_train['Total_Emissions'] - df_train['Predicted_Emissions']

# Calculate absolute residuals
df_train['Absolute_Residuals'] = df_train['Residuals'].abs()

# Sort DataFrame by absolute residuals
df_train_sorted = df_train.sort_values('Absolute_Residuals', ascending=False)

# Display the top 10 observations with the largest residuals
print(df_train_sorted.head(10))
