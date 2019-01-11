from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
#input

#imputing

# standartize X


X = pd.DataFrame(MinMaxScaler().fit_transform(X))

variance_cutoff = VarianceThreshold(threshold=0.5)
variance_cutoff.fit_transform(X)

#pca
pca = PCA(n_components=2)

#rfc
rfc = RandomForestClassifier()
model = SelectFromModel(rfc, prefit=True)
