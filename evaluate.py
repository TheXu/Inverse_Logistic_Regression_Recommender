# -*- coding: utf-8 -*-
"""
Inverse Logistic Regression Recommender
Created on 2019

@author: Alex Xu <ayx2@case.edu>
"""
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display


def _error_metrics_(column, y_true, y_pred):
    """
    Compute Regression Error metrics on the specified column

    Parameters
    ----------
    column : str
        Column to calculate statistics on

    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    col_metrics : pandas.dataframe
        RMSE, MAE, and R-Squared calculated for the column
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    col_metrics = pd.DataFrame({column:[rmse, mae, r2]},
        index=['RMSE', 'MAE', 'R2'])
    return(col_metrics)


def validate(recommender_class):
    """
    Validate the accuracy of inverse linear model recommendations by
    calculating the approximation recommendations and comparing it against
    the original dataset

    Parameters
    ----------
    recommender_class : class
        Inverse Linear Model Recommender Class

    Attributes
    ----------
    approximation : pandas.dataframe
        Approximation of original dataset using predict_df method from
        recommender_class

    column_metrics : pandas.dataframe
        Evaluation statistics computed for each column

    all_metrics: pandas.dataframe
        Evaluation statistics using each value of dataset
    """
    # Compute approximation for entire dataset using true target
    predict_df = getattr(recommender_class, 'predict_df')
    approximation = predict_df(original_target=True, rows='all', columns='all')
    # Assign to attribute of recommender class
    setattr(recommender_class, 'approximation', approximation)
    print('\nAssigned approximation dataframe to the recommender class.')
    # Set columns to validate
    target = getattr(recommender_class, 'y')
    columns_to_validate = getattr(recommender_class, 'df').columns.tolist()
    columns_to_validate.remove(target)
    ##### Compute evaluation metrics by column
    metrics_by_col = list(map(lambda col: _error_metrics_(col,
        getattr(recommender_class, 'df')[col],
        getattr(recommender_class, 'approximation')[col]),
        columns_to_validate))
    metrics_by_col = pd.concat(metrics_by_col, axis=1)
    # Assign column metrics to attribute of recommender class
    setattr(recommender_class, 'column_metrics', metrics_by_col)
    print('\nAssigned column_metrics dataframe to the recommender class.')
    display(recommender_class.column_metrics)
    ##### Compute evaluation metrics for every value approximated value in the
    # dataset
    # Put original and approximation datasets into series
    original_series = pd.concat(list(map(lambda col: getattr(recommender_class
        , 'df')[col], columns_to_validate)), axis=0)
    approx_series = pd.concat(list(map(lambda col: getattr(recommender_class
        , 'approximation')[col], columns_to_validate)), axis=0)
    all_metrics = _error_metrics_('all_values', original_series, approx_series)
    # Assign column metrics to attribute of recommender class
    setattr(recommender_class, 'all_metrics', all_metrics)
    print('\nAssigned all_metrics dataframe to the recommender class.')
    print('Beware of difference of magnitudes and scales when using all elements in dataset')
    display(recommender_class.all_metrics)
