#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29, 2023

Climate Change Risk and Opportunity Analysis

Author: Anuj Patil

Description: This script is designed to train a Random Forest model to predict emissions based
on processed and encoded data. It ensures that the training and target datasets are properly
aligned, proceeds to model fitting and evaluation, and saves the predictions, performance metrics,
and the trained model for later use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # for saving the model

def main():
    try:
        # Load the data
        df = pd.read_csv('data/cleaned-data-label-encoded.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("File not found. Please ensure the 'cleaned-data-label-encoded.csv' file is in the correct directory.")
        return

    try:
        # Separate the target variable from the features
        X = df.drop(columns=['Total_Emissions'])
        y = df['Total_Emissions']
    except KeyError:
        print("Column 'Total_Emissions' does not exist in the DataFrame. Please check the column names.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) != len(y_train):
        raise ValueError("The number of samples in X_train and y_train does not match.")

    # Model creation and training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Output the performance metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")

    # Save the model
    joblib.dump(model, 'models/random_forest_model.joblib')
    print("Model saved to 'models/random_forest_model.joblib'.")

    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predictions_df.to_csv('results/predictions.csv', index=False)
    print("Predictions saved to 'results/predictions.csv'.")

    # Save performance metrics
    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv('results/performance_metrics.csv', index=False)
    print("Performance metrics saved to 'results/performance_metrics.csv'.")

if __name__ == '__main__':
    main()
