from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LSASCutoff (BaseEstimator, TransformerMixin):
    def __init__(self, threshold, lsas_column_location=1):
        self.threshold = threshold
        self.lsas_column_location = lsas_column_location

    def fit(self, X, Y):
        return self

    def transform(self, X, Y):
        if self.threshold is None:
            return Y
        copy_y = np.where(X[self.lsas_column_location] > self.threshold, 1, 0)
        return copy_y
