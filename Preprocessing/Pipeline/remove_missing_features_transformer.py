from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RemoveMissingFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, Y=None):
        self.is_missing = X.isnull().values.any(axis=0)
        return self

    def transform(self, X, Y=None):
        copy_x = pd.DataFrame(X)
        self.is_missing += copy_x.isnull().values.any(axis=0)

        copy_x = copy_x.iloc[:, ~self.is_missing]

        return copy_x.values
