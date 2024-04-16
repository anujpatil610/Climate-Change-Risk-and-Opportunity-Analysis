#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:23:24 2023

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler

def main():
    try:
        # Load the data
        df = pd.read_csv('cleaned-data-label-encoded.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("File not found. Please ensure the 'cleaned-data-label-encoded.csv' file is in the same directory as this script.")
        return

    try:
        # Separate the target variable from the features
        X = df.drop(columns=['Total_Emissions'])
        y = df['Total_Emissions']
    except KeyError:
        print("Column names in the dataset do not match expected names ('Total_Emissions'). Please check the dataset columns.")
        return

    # Data Preprocessing: Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Capture the indices of the test set which you'll need later
    X_test_indices = X_test.index

    # Lasso Regression for feature selection
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    lasso_feature_selection = SelectFromModel(lasso, prefit=True)
    selected_features = X.columns[lasso_feature_selection.get_support()]

    # Print selected features from Lasso Regression
    print("Selected Features from Lasso Regression:")
    print(selected_features)

    # Save the selected features to a CSV file
    selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
    selected_features_df.to_csv('selected_features.csv', index=False)
    print("Selected features saved to 'selected_features.csv'.")

    # Keep only the selected features in the dataset
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Normalize the selected features
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train_selected)
    X_test_normalized = scaler.transform(X_test_selected)

    # Save the normalized feature data to CSV files
    X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=selected_features)
    X_test_normalized_df = pd.DataFrame(X_test_normalized, columns=selected_features)
    X_train_normalized_df.to_csv('X_train_normalized.csv', index=False)
    X_test_normalized_df.to_csv('X_test_normalized.csv', index=False)
    print("Normalized data saved to CSV files.")

if __name__ == '__main__':
    main()

