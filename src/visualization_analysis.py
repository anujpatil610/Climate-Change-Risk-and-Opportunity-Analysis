#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 5, 2023

Climate Change Risk and Opportunity Analysis

Author: Anuj Patil

Description: This script visualizes the predictions from the RandomForest model used to predict emissions,
provides analysis through visual graphs including scatter plots of predictions vs actuals, error histograms, and
feature importance charts. Additional plots include density plots and boxplots for deeper analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for loading the model
import numpy as np
import os

def load_data():
    """Load the actual and predicted emissions data for visualization."""
    try:
        actual = pd.read_csv('data/y_test.csv')
        predictions = pd.read_csv('results/predictions.csv')
        return actual['Total_Emissions'], predictions['Predicted']
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit()
    except KeyError as e:
        print(f"Column missing in data: {e}")
        exit()

def save_figure(plt, filename):
    """Save the plot to the specified file."""
    graph_dir = 'graphs'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    plt.savefig(f"{graph_dir}/{filename}")
    print(f"Graph saved to {graph_dir}/{filename}")

def plot_actual_vs_predicted(actual, predictions):
    """Plot actual vs predicted values in a scatter plot to visualize the accuracy of predictions."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual, y=actual, color="blue", label='Actual Emissions')
    sns.scatterplot(x=actual, y=predictions, color="red", label='Predicted Emissions')
    plt.title('Actual vs Predicted Emissions')
    plt.xlabel('Actual Emissions')
    plt.ylabel('Emissions Value')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', label='Ideal Prediction Line')
    plt.legend()
    save_figure(plt, 'actual_vs_predicted.png')
    plt.show()

def plot_error_distribution(actual, predictions):
    """Plot the distribution of prediction errors."""
    errors = predictions - actual
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    save_figure(plt, 'error_distribution.png')
    plt.show()

def plot_density_comparison(actual, predictions):
    """Plot density comparison of actual vs predicted emissions."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(actual, label='Actual Emissions', fill=True)
    sns.kdeplot(predictions, label='Predicted Emissions', fill=True)
    plt.title('Density Plot of Actual vs Predicted Emissions')
    plt.xlabel('Emissions')
    plt.legend()
    save_figure(plt, 'density_comparison.png')
    plt.show()

def plot_box_comparison(actual, predictions):
    """Plot boxplot comparison of actual vs predicted emissions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[actual, predictions], palette=["blue", "red"], width=0.5)
    plt.xticks([0, 1], ['Actual Emissions', 'Predicted Emissions'])
    plt.title('Boxplot of Actual vs Predicted Emissions')
    plt.ylabel('Emissions')
    save_figure(plt, 'boxplot_comparison.png')
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
    save_figure(plt, 'feature_importance.png')
    plt.show()

def main():
    actual, predictions = load_data()
    model = joblib.load('models/random_forest_model.joblib')  # Load the fitted model
    plot_actual_vs_predicted(actual, predictions)
    plot_error_distribution(actual, predictions)
    plot_density_comparison(actual, predictions)
    plot_box_comparison(actual, predictions)
    plot_feature_importance(model)

if __name__ == '__main__':
    main()
