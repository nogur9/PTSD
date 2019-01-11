import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def save_feature_importance(path,feature_importance, feature_names):
    print(feature_importance, feature_names)
    with open(os.path.join(path, "feature_importance.txt"), "w") as f:
        for i,j in zip(feature_importance, feature_names):
            f.write(str(i) + ' ' + str(j) + "\n")
    feature_importance = sorted(feature_importance, reverse=True)
    plt.plot(range(len(feature_importance)), feature_importance)
    plt.show()

def get_k_most_important_features(X,Y, k , feature_importance):
    importance_zip = sorted(zip(feature_importance,X), key=lambda x: x[0])
    include_columns = [importance_zip[i][1] for i in range(k)]
    highest_features = pd.concat([X[include_columns],Y], axis=1)
    highest_features.columns = [str(i) for i in range(k)]+['group']
    return highest_features, X[include_columns]

def get_k_most_important_features_after_pca(X,Y, k , feature_importance):
    importance_zip = sorted(zip(feature_importance,X.T), key=lambda x: x[0])
    include_columns = [importance_zip[i][1] for i in range(k)]
    highest_features = pd.concat([pd.DataFrame(include_columns).T,Y], axis=1)
    highest_features.columns = [str(i) for i in range(k)]+['group']
    return highest_features

def get_feature_importance(X, Y):
    model = RandomForestClassifier()
    model.fit(X, Y)
    return model.feature_importances_