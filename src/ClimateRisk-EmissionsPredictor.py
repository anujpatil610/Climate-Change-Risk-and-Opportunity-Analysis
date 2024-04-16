#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29, 2023

Climate Change Risk and Opportunity Analysis

Author: Anuj Patil

Description: This script trains a RandomForest model to predict emissions, ensures datasets
are properly aligned, saves the model, predictions, performance metrics, and notably the test dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # for saving the model
import os

def load_data(filepath):
    """ Load data from a CSV file """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully from:", filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Please check your path.")
        exit()

def prepare_data(df):
    """ Prepare data by separating features and target and performing a train-test split """
    X = df.drop(columns=['Total_Emissions'])
    y = df['Total_Emissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """ Train a RandomForestRegressor """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_and_save(model, X_test, y_test):
    """ Evaluate the model and save the outputs including the model and performance metrics """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}, Mean Absolute Error (MAE): {mae}, R-squared: {r2}")

    # Saving outputs
    joblib.dump(model, 'models/random_forest_model.joblib')
    print("Model saved at 'models/random_forest_model.joblib'.")

    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('results/predictions.csv', index=False)
    print("Predictions saved at 'results/predictions.csv'.")

    metrics_df = pd.DataFrame({'MSE': [mse], 'MAE': [mae], 'R2': [r2]})
    metrics_df.to_csv('results/performance_metrics.csv', index=False)
    print("Performance metrics saved at 'results/performance_metrics.csv'.")

    y_test.to_csv('data/y_test.csv', index=False)
    print("Test dataset saved at 'data/y_test.csv'.")

def main():
    data_filepath = 'data/cleaned-data-label-encoded.csv'
    df = load_data(data_filepath)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_and_save(model, X_test, y_test)

if __name__ == '__main__':
    main()
