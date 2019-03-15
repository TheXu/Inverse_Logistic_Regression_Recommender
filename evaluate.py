# -*- coding: utf-8 -*-
"""
Inverse Logistic Regression Recommender
Created on 2019

@author: Alex Xu <ayx2@case.edu>
"""
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def _error_metrics_(column, y_true, y_pred):
    """
    Compute Regression Error metrics on the specified column
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(column, y_true, y_pred)
    col_metrics = pd.DataFrame({column:[rmse, mae]}, index=['RMSE', 'MAE'])
    return(col_metrics)


def evaluate(recommender_class):
    # Compute approximation for entire dataset using true target
    predict_df = getattr(recommender_class, 'predict_df')
    approximation = predict_df(original_target=True, rows='all', columns='all')
    # Assign to attribute of recommender class
    setattr(recommender_class, 'approximation', approximation)
    print('\nAssigned approximation as an attribute of the recommender class.')
    ##### Compute evaluation metrics by column
    metrics_by_col = list(map(lambda col: _error_metrics_(
        getattr(recommender_class, 'df')[col],
        getattr(recommender_class, 'approximation')[col]),
        getattr(recommender_class, 'df').columns.tolist()))
    metrics_by_col = pd.concat(metrics_by_col, axis=1)
    # Assign column metrics to attribute of recommender class
    setattr(recommender_class, 'column_metrics', metrics_by_col)
    ##### Compute evaluation metrics for every value approximated value in the
    # dataset