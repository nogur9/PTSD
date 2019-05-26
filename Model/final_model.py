from fancyimpute import IterativeImputer
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFpr, SelectKBest, SelectFdr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import numpy as np
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from Model.model_object import Model
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from feature_engineering.engineering import FeatureEngineering

targets = {
    'intrusion': 0,
    'avoidance': 0,
    'hypertension': 0,
    'depression': 0,
    'only_avoidance': 0,
    'PCL_Strict3': 1,
    'regression_cutoff_33': 0,
    'regression_cutoff_50': 0,
    'tred_cutoff': 0,
}

pipeline_per_target = {
    'intrusion':
        Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.05)),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', XGBClassifier(n_estimators=300, max_depth=5))]),
    'avoidance':
        Pipeline(steps=[
                ('feature_selection',  RFE(estimator=XGBClassifier(scale_pos_weight=5.88, n_estimators=100),
                                           n_features_to_select=20)),
                ('classifier', BalancedRandomForestClassifier(n_estimators=300, max_depth=10))]),
    'hypertension':
        Pipeline(steps=[
            ('feature_selection', RFE(estimator=XGBClassifier(n_estimators=100, scale_pos_weight=3.51),
                                      n_features_to_select=20)),
            ('sampling', SMOTE(k_neighbors=10)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=100))]),
    'depression':
        Pipeline(steps=[
            ('feature_selection', SelectFdr(alpha=0.1)),
            ('sampling', SMOTE(k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100))]),
    'only_avoidance':
        Pipeline(steps=[
            ('feature_selection', RFE(XGBClassifier(n_estimators=100, max_depth=3), n_features_to_select=10)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=500, max_depth=10))]),
    'PCL_Strict3':
        Pipeline(steps=[
            ('feature_selection', SelectKBest(k=20)),
            ('sampling', SMOTE(k_neighbors=5)),
            ('classifier', XGBClassifier(max_depth=3, n_estimators=100))]),
    'regression_cutoff_33':
        Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.033)),
            ('sampling', SMOTE(k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5))]),
    'regression_cutoff_50':
        Pipeline(steps=[
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', SMOTE(k_neighbors=10)),
            ('classifier', XGBClassifier(max_depth=2, n_estimators=100))]),
    'tred_cutoff':
        Pipeline(steps=[
            ('feature_selection', SelectKBest(k=20)),
            ('sampling', SMOTE(k_neighbors=10)),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=2))])
}
class TargetEnsembler(object):

    def __init__(self, features, use_feature_engineering=1, train_on_partial_prediction=1):
        self.features = features
        self.use_feature_engineering = use_feature_engineering
        self.train_on_partial_prediction = train_on_partial_prediction
        self.combined_model = None
        self.trained_pipelines = {
            'intrusion': None,
            'avoidance': None,
            'hypertension': None,
            'depression': None,
            'only_avoidance': None,
            'PCL_Strict3': None,
            'regression_cutoff_33': None,
            'regression_cutoff_50': None,
            'tred_cutoff': None
         }

        # create list of targets
        self.targets_list = [i for i in targets if targets[i] == 1]

    def fit(self, X_train, y_train):

        predictions_list = []

        for target in self.targets_list:
            if self.use_feature_engineering:
                X = FeatureEngineering(X_train[self.features], "avoidance_cutoff").engineer_features().values
            else:
                X = X_train[self.features].values

            if target == "PCL_Strict3":
                y = y_train["PCL_Strict3"].apply(lambda x: int(x))
            else:
                y = X_train["avoidance_cutoff"].apply(lambda x: int(x))

            pipeline = pipeline_per_target[target]
            scores = cross_val_score(pipeline, X, y, scoring='f1', cv=StratifiedKFold(10))
            print(f"{target} - {sum(scores)/len(scores)}")

            if self.train_on_partial_prediction:
                _X_train, _X_test, _y_train, _y_test = train_test_split(X, [y, y_train["PCL_Strict3"].apply(lambda x: int(x))], test_size=0.33)
                self.trained_pipelines[target] = pipeline.fit(_X_train, _y_train)
                y_pred = self.trained_pipelines[target].predict(_X_test)
                predictions_list.append(y_pred)
                print("test f1", target, f1_score(_y_test, y_pred))
                self.trained_pipelines[target] = pipeline.fit(X, y)

            else:
                self.trained_pipelines[target] = pipeline.fit(X, y)
                predictions_list.append([self.trained_pipelines[target].predict(X)])

        X = predictions_list
        y = y_train["PCL_Strict3"]
        self.combined_model = LogisticRegression(penalty='l1').fit(np.array(X).reshape(-1, len(predictions_list)), y)

    def predict(self, X_test):

        predictions_list = []
        for target in self.targets_list:
            if self.use_feature_engineering:
                X = FeatureEngineering(X_test[self.features], "avoidance_cutoff").engineer_features().values
            else:
                X = X_test[self.features].values
            predictions_list.append([self.trained_pipelines[target].predict(X)])

        X = predictions_list
        y_pred = self.combined_model.predict(np.array(X).reshape(-1, len(predictions_list)))
        return y_pred


def cv(X_train, y_train, features):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    scores_f = []
    scores_p = []
    scores_r = []

    for train, test in kfold.split(X_train, y_train):
        model = TargetEnsembler(features)

        X_train_cv = pd.DataFrame(X_train.values[train], columns=X_train.columns)
        y_train_cv = pd.DataFrame(y_train.values[train], columns=["PCL_Strict3"])
        X_test_cv = pd.DataFrame(X_train.values[test], columns=X_train.columns)
        y_test_cv = pd.DataFrame(y_train.values[test], columns=["PCL_Strict3"])
        model.fit(X_train_cv, y_train_cv)

        y_pred = model.predict(X_test_cv)

        s_f = f1_score(y_test_cv, y_pred)
        s_p = precision_score(y_test_cv, y_pred)
        s_r = recall_score(y_test_cv, y_pred)
        print("\tscores f1", (s_f))
        print("\tscores p", (s_p))
        print("\tscores r", (s_r))
        scores_f.append(s_f)
        scores_p.append(s_p)
        scores_r.append(s_r)

    print("mean scores f1", np.mean(scores_f))
    print("mean scores p", np.mean(scores_p))
    print("mean scores r", np.mean(scores_r))


def runner():
    print(*[f"{target}_{targets[target]}" for target in targets])

    m = Model()
    X = m.df.drop("PCL_Strict3", axis=1)
    Y = pd.DataFrame(m.df["PCL_Strict3"], columns=["PCL_Strict3"])
    features = m.features + m.features_2
    cv(X, Y, features)

    model = TargetEnsembler(features)
    model.fit(X, Y)

    y_pred = model.predict(m.X_test)
    s_f = f1_score(m.y_test, y_pred)
    s_p = precision_score(m.y_test, y_pred)
    s_r = recall_score(m.y_test, y_pred)
    print("test f1", s_f)
    print("test recall", s_r)
    print("test precision", s_p)


runner()
