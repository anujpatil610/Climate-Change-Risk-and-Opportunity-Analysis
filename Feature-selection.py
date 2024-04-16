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

# Load the data from an Excel file (change 'your_data.xlsx' to your actual file path)
df = pd.read_csv('cleaned_data_label_encoded.csv')

# Target variable is named 'total_emission'
# Separate the target variable from the features
X = df.drop(columns=['Total_Emissions'])
y = df['Total_Emissions']

# Data Preprocessing: Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets (optional, for Lasso regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso Regression
lasso = Lasso(alpha=0.1)  # You can experiment with different values of alpha
lasso.fit(X_train, y_train)

lasso_feature_selection = SelectFromModel(lasso, prefit=True)
selected_features = X.columns[lasso_feature_selection.get_support()]

# 'selected_features' contains the features selected by Lasso regression

print("Selected Features from Lasso Regression:")
print(selected_features)

# Store the selected features in a CSV file
# selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
# selected_features_df.to_csv('selected_features.csv', index=False)

# Keep only the selected features in the dataset
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Perform normalization on the selected features
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_selected)
X_test_normalized = scaler.transform(X_test_selected)

# Save the normalized feature data to CSV files
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=selected_features)
X_test_normalized_df = pd.DataFrame(X_test_normalized, columns=selected_features)

# Save the data to CSV files
X_train_normalized_df.to_csv('X_train_normalized.csv', index=False)
X_test_normalized_df.to_csv('X_test_normalized.csv', index=False)

#%%

# Read the Excel file into a DataFrame
df = pd.read_excel('cleaned_data.xlsx')

# Convert date column to datetime format
df['date_column'] = pd.to_datetime(df['date_column'])

# Convert float column to numeric (replace 'float_column' with the actual column name)
df['float_column'] = pd.to_numeric(df['float_column'], errors='coerce')

# Drop rows with NA values in float column, if needed
df = df.dropna(subset=['float_column'])

# Now, you can proceed with feature selection or other data analysis tasks
