#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 4 19:22:5 2023

Climate Change Risk and Opportunity Analysis

Authors:Anuj Patil

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    X_train = pd.read_csv('../results/X_train_normalized.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze('columns')

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predicting and saving results
    y_pred = model.predict(X_train)  # Assuming X_train for simplicity
    pd.DataFrame(y_pred, columns=['Predicted_Emissions']).to_csv('../results/optimal_y_predictions.csv', index=False)

if __name__ == '__main__':
    main()

