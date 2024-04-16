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
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    X_test_indices = pd.read_csv('results/X_test_indices.csv')
    encoded_data = pd.read_csv('data/cleaned-data-label-encoded.csv')

    # Placeholder for predictions, replace it with actual predictions loaded from your model's output
    optimal_y_pred_test = [10] * len(X_test_indices)  # Example static data

    df_results = pd.DataFrame({
        'Organization': encoded_data.loc[X_test_indices['index'], 'Organization'],
        'Primary sector': encoded_data.loc[X_test_indices['index'], 'Primary sector'],
        'Country': encoded_data.loc[X_test_indices['index'], 'Country'],
        'Predicted_Emissions': optimal_y_pred_test
    })

    top_10_companies = df_results.sort_values('Predicted_Emissions', ascending=False).head(10)
    top_10_sectors = df_results.groupby('Primary sector').sum().sort_values('Predicted_Emissions', ascending=False).head(10)

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

if __name__ == '__main__':
    main()

