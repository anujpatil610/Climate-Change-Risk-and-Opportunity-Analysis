#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Change Risk and Opportunity Analysis

Created on Sat Aug 5, 2023
Author: Anuj Patil

This script performs data analysis on predicted emissions across companies and sectors,
visualizes the top contributors, and calculates key performance metrics.
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

def load_data(filepath):
    """Load and return the encoded dataset from a specified filepath."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        exit()

def main():
    encoded_data = load_data('cleaned-data-label-encoded.csv')

    # Assume X_test_indices and optimal_y_pred_test are defined; if not, define them here.
    # Example:
    # X_test_indices = [indices]
    # optimal_y_pred_test = [predictions]

    # Create a new DataFrame that includes company name, sector, and predicted emissions
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

    # Visualization
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

    # Performance Metrics
    mse = mean_squared_error(y_test, optimal_y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, optimal_y_pred_test)
    r2 = r2_score(y_test, optimal_y_pred_test)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")

if __name__ == '__main__':
    main()
