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
    df = pd.read_csv('data/cleaned-data-label-encoded.csv')
    X = df.drop(columns=['Total_Emissions'])
    y = df['Total_Emissions']

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    model = SelectFromModel(lasso, prefit=True)
    selected_features = X.columns[model.get_support()]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train_selected)
    X_test_normalized = scaler.transform(X_test_selected)

    # Save necessary outputs for other scripts
    pd.DataFrame(X_train_normalized, columns=selected_features).to_csv('results/X_train_normalized.csv', index=False)
    pd.DataFrame(X_test_normalized, columns=selected_features).to_csv('results/X_test_normalized.csv', index=False)
    pd.DataFrame(X_test.index, columns=['index']).to_csv('results/X_test_indices.csv', index=False)

if __name__ == '__main__':
    main()


