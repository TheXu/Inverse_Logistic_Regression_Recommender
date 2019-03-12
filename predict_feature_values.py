# -*- coding: utf-8 -*-
"""
Inverse Logistic Regression Recommender
Created on 2019

@author: Alex Xu <ayx2@case.edu>
"""
import numpy as np
import pandas as pd

class inverse_logistic_regression_recommender:
    """
    Predict or recommend a select feature value, so that it can be classified
    as a select class. Based on binary logistic regression coefficients, other
    feature values.
    
    Parameters
    ----------
    df : pandas.dataframe

    y : str
        Dependent or Target Variable with binary positive and negative
        class

    coefs : list
        List of floats corresponding to the logistic regression binary
        classification coefficients. Should correspond with non-target 
        column names of "df", i.e. Feature Column 0's regression
        coefficient is coefs[0], and so on.

    Attributes
    ----------
    df : pandas.dataframe
        From __init__

    y : str
        From __init__

    coefs : list
        From __init__

    mean_feature_values : pandas.dataframe
        Mean feature values by target variable class

    interim_logits : list
        Interim logits created by mean feature values matrix multiplied by
        logistic regression coefficients. Used to create distinction between
        classes to be used by recommender
    """
    def __init__(self, df, y, coefs):
        self.df = df
        self.y = y
        self.coefs = coefs
        # Compute mean feature values by dependent variable class
        self.mean_feature_values = df.groupby(y).mean()
        # Compute interim logits using mean feature values of each dependent
        # variable class
        self.interim_logits = self.mean_feature_values.\
            apply(lambda row: np.dot(row, coefs), axis=1).tolist()

    def predict(self, predict_column, target_class, feature_values):
        """
        Predict minimum/maximum column value, given a row of other fixed
        feature values, to reach a desired class

        Parameters
        ----------
        predict_column : str
            Name of feature value column to recommend and predict on

        target_class : bool
            Representing desired class, which 'predict_column' will be tuned
            for. 1 or True is positive label. 0 or False is negative label.

        feature_values : list
            List of numerics representing fixed feature values, which
            'predict_column' and 'target_class' will predict on

        Returns
        -------
        prediction : float
            Predicted column value to achieve desired class, when all other
            features are held constant
        """
        # Create a copy of coefficients list
        coefs = self.coefs.copy()
        # Get index position of column to predict on by feature values
        # self.df column names
        predict_column_index = self.df.drop(self.y, axis=1).columns.\
            tolist().index(predict_column)
        # Extract predict column coefficient, and remove it from feature
        # value coefficients
        predict_column_coef = coefs[predict_column_index]
        del coefs[predict_column_index]
        # Compute prediction of feature value given interim logits for
        # specified class, feature values
        prediction = (self.interim_logits[target_class] - np.dot(feature_values, coefs))/predict_column_coef
        return(prediction)


    def predict_df(self):
        """
        Predict all dataset feature values using inverse logistic
        regression classification to achieve opposite target class labels,
        using true feature values.

        Parameters
        ----------
        self

        Returns
        -------
        approximation : pandas.dataframe
            Prediction/Recommendation of every feature value, given original
            other feature values, to achieve opposite-of-original class labels
        """
        # Create function that approximates feature values by the row
        def predict_row(self, row):
            # Iterate over all self.df column names
            # If column iterated is target variable, then return 
            # opposite to true target value
            target = int(not int(row[self.y]))
            col_predict = list(map(lambda col: target if col==self.y
            # Else, run ilrc recommender to approximate features, based on
            # Other features and opposite to true class labels
                            else inverse_logistic_regression_recommender.\
                                predict(self, col, target, row.loc[(row.index != col) & (row.index != self.y)].tolist()), self.df.columns.tolist()))
            return(col_predict)
        # Iterate function above over rows of dataset
        approximation = pd.DataFrame(self.df.\
            apply(lambda row: predict_row(self, row), axis=1).\
            tolist())
        approximation.columns = self.df.columns.tolist()
        return(approximation)


    def approximate_df(self):
        """
        Approximate all dataset feature values using inverse logistic
        regression classification. Based on true target class labels, and
        true feature values. Could be used to evaluate accuracy of
        recommendations

        Parameters
        ----------
        self

        Returns
        -------
        approximation : pandas.dataframe
            Approximation/Recommendation of every feature value, given original
            other feature values, to achieve original class labels
        """
        # Create function that approximates feature values by the row
        def predict_row(self, row):
            # Iterate over all self.df column names
            # If column iterated is target variable, then return target value
            col_predict = list(map(lambda col: int(row[self.y]) if col==self.y
            # Else, run ilrc recommender to approximate features, based on
            # Other features and true class labels
                            else inverse_logistic_regression_recommender.\
                                predict(self, col, int(row[self.y]), row.loc[(row.index != col) & (row.index != self.y)].tolist()), self.df.columns.tolist()))
            return(col_predict)
        # Iterate function above over rows of dataset
        approximation = pd.DataFrame(self.df.\
            apply(lambda row: predict_row(self, row), axis=1).\
            tolist())
        approximation.columns = self.df.columns.tolist()
        # Store dataset within class, to use for evaluation
        self.approximation = approximation
        return(approximation)