from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RemoveInvalidColumns (BaseEstimator, TransformerMixin):
    def __init__(self, lsas_column_location=1, subject_number_column_location=0):
        self.lsas_column_location = lsas_column_location
        self.subject_number_column_location = subject_number_column_location
    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        copy_x = np.delete(X, [self.lsas_column_location, self.subject_number_column_location], axis=1)
        return copy_x
