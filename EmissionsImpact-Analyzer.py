#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 5 24:22:5 2023

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

# Create a new DataFrame that includes company name, sector and predicted emissions
df_results = pd.DataFrame({
    'Organization': encoded_data.loc[X_test_indices, 'Organization'], 
    'Primary sector': encoded_data.loc[X_test_indices, 'Primary sector'], 
    'Country': encoded_data.loc[X_test_indices, 'Country'], 
    'Predicted_Emissions': optimal_y_pred_test
})

# Get the top 10 companies with the highest predicted emissions
top_10_companies = df_results.sort_values('Predicted_Emissions', ascending=False).head(10)

# Get the top 10 sectors with the highest predicted emissions
top_10_sectors = df_results.groupby('Primary sector').sum().sort_values('Predicted_Emissions', ascending=False).head(10)

# Visualize the results
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_companies, x='Organization', y='Predicted_Emissions')
plt.title('Top 10 Companies with Highest Predicted Emissions')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_sectors, x='Country', y='Predicted_Emissions')
plt.title('Top 10 Sectors with Highest Predicted Emissions')
plt.xticks(rotation=90)
plt.show()


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, optimal_y_pred_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, optimal_y_pred_test)

# Calculate R-squared
r2 = r2_score(y_test, optimal_y_pred_test)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
