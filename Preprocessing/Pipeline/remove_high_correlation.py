from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class RemoveCorrelationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.7, pca_components_ratio=3):
        self.correlation_threshold = correlation_threshold
        self.pca_components_ratio = pca_components_ratio


    def fit(self, X, Y=None):
        df = pd.DataFrame(X)
        df_corr = df.corr(method='pearson')
        df_corr = df_corr - np.eye(df.shape[1])
        outliares_corr = df_corr[np.abs(df_corr) > self.correlation_threshold]
        self.outliares_corr = outliares_corr.dropna(axis=1, how='all')

        correlated_df = df[self.outliares_corr.columns]

        n_components = len(self.outliares_corr.columns) // self.pca_components_ratio
        pca = PCA(n_components=n_components)

        correlated_df = pca.fit_transform(correlated_df)
        self.correlated_df = pd.DataFrame(correlated_df, columns=["pca_{}".format(i) for i in range(n_components)])

        return self

    def transform(self, X, Y=None):
        df = pd.DataFrame(X)
        df = df.drop((self.outliares_corr.columns), axis=1)
        df = df.join(self.correlated_df)
        return df

# cutoff high correlation
