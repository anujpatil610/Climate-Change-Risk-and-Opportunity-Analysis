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
import joblib  # for loading the model
import numpy as np

def load_data():
    """Load the actual and predicted emissions data for visualization."""
    try:
        actual = pd.read_csv('data/y_test.csv')  # Ensure this contains the target column 'Total_Emissions'
        predictions = pd.read_csv('results/predictions.csv')  # Ensure this contains 'Predicted_Emissions'
        return actual['Total_Emissions'], predictions['Predicted']
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit()
    except KeyError as e:
        print(f"Column missing in data: {e}")
        exit()


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

def plot_feature_importance(model):
    """Plot the feature importance from the RandomForest model."""
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
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
    model = joblib.load('models/random_forest_model.joblib')  # Load the fitted model

    plot_actual_vs_predicted(actual, predictions)
    plot_error_distribution(actual, predictions)
    plot_feature_importance(model)

if __name__ == '__main__':
    main()
