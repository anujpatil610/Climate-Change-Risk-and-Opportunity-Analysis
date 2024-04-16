#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 5, 2023

Climate Change Risk and Opportunity Analysis

Author: Anuj Patil

Description: This script visualizes the predictions from the RandomForest model used to predict emissions,
provides analysis through visual graphs including scatter plots of predictions vs actuals, error histograms, and
feature importance charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_data():
    """Load the actual and predicted emissions data for visualization."""
    actual = pd.read_csv('data/y_test.csv')  # Adjust path as needed
    predictions = pd.read_csv('results/predictions.csv')  # Adjust path as needed
    return actual, predictions

def plot_actual_vs_predicted(actual, predictions):
    """Plot actual vs predicted values in a scatter plot to visualize the accuracy of predictions."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual, y=predictions)
    plt.title('Actual vs Predicted Emissions')
    plt.xlabel('Actual Emissions')
    plt.ylabel('Predicted Emissions')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--')
    plt.show()

def plot_error_distribution(actual, predictions):
    """Plot the distribution of prediction errors."""
    errors = predictions - actual
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plot the feature importance from the RandomForest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

def main():
    actual, predictions = load_data()
    model = RandomForestRegressor()  # Assuming the model is already fitted and saved, load it if not
    feature_names = ['feature1', 'feature2', 'feature3', 'etc']  # Replace with actual feature names

    plot_actual_vs_predicted(actual['Total_Emissions'], predictions['Predicted_Emissions'])
    plot_error_distribution(actual['Total_Emissions'], predictions['Predicted_Emissions'])
    plot_feature_importance(model, feature_names)

if __name__ == '__main__':
    main()
