from fancyimpute import IterativeImputer
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFpr, SelectKBest, SelectFdr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from Model.model_object import Model
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from feature_engineering.engineering import FeatureEngineering
from sklearn.model_selection import LeaveOneOut

check_on_test_set = 1
targets_dict = {
    'intrusion_cutoff': 0,
    'avoidance_cutoff': 0,
    'hypertention_cutoff': 0,
    'depression_cutoff': 0,
    'only_avoidance_cutoff': 0,
    'PCL_Strict3': 1,
    'regression_cutoff_33': 1,
    'regression_cutoff_50': 1,
    'tred_cutoff': 0,
}

pipeline_per_target = {
    'intrusion_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', SMOTE(k_neighbors=10)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ]),
    'avoidance_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ]),
    'hypertention_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=3.5, reg_alpha=1), n_features_to_select=10)),
            ('sampling', SMOTE(k_neighbors=5)),
            ('classifier', SVC(C=0.5, kernel='linear'))
        ]),
    'depression_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(k_neighbors=5)),
            ('classifier', RandomForestClassifier(max_depth=3, n_estimators=100))
        ]),
    'only_avoidance_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=0.1))
        ]),
    'PCL_Strict3':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.2)),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=30, penalty='l1'))
        ]),
    'regression_cutoff_33':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.2)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=7)),
            ('classifier', RandomForestClassifier(n_estimators=600, max_depth=5))
        ]),
    'regression_cutoff_50':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ]),
    'tred_cutoff':
        Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
        ])
}

class TargetEnsembler(object):

    def __init__(self, features, use_feature_engineering=0, train_on_partial_prediction=0, use_and_func=1,
                 check_on_test_set=0, X_test=None, y_test=None):
        self.features = features
        self.use_feature_engineering = use_feature_engineering
        self.train_on_partial_prediction = train_on_partial_prediction
        self.use_and_func = use_and_func
        self.check_on_test_set = check_on_test_set
        self.X_test = X_test
        self.y_test=y_test
        self.combined_model = None
        self.trained_pipelines = {
            'intrusion_cutoff': None,
            'avoidance_cutoff': None,
            'hypertention_cutoff': None,
            'depression_cutoff': None,
            'only_avoidance_cutoff': None,
            'PCL_Strict3': None,
            'regression_cutoff_33': None,
            'regression_cutoff_50': None,
            'tred_cutoff': None
         }

        # create list of targets
        self.targets_list = [i for i in targets_dict if targets_dict[i] == 1]

    def fit(self, X_train, y_train):

        predictions_list = []

        for target in self.targets_list:
            if self.use_feature_engineering:
                X = FeatureEngineering(X_train[self.features], target).engineer_features().values
            else:
                X = X_train[self.features].values

            if target == "PCL_Strict3":
                y = y_train[target].apply(lambda x: int(x))
            else:
                y = X_train[target].apply(lambda x: int(x))

            pipeline = pipeline_per_target[target]
#            scores = cross_val_score(pipeline, X, y, scoring='f1', cv=StratifiedKFold(5))
#            print(f"{target} - {sum(scores)/len(scores)}")

            if self.train_on_partial_prediction:
                combined_y = pd.DataFrame(y, columns=[target])
                if target != "PCL_Strict3":
                    combined_y["PCL_Strict3"] = y_train["PCL_Strict3"].apply(lambda x: int(x))

                _X_train, _X_test, _y_train, _y_test = \
                    train_test_split(X, combined_y, test_size=0.25)
                self.trained_pipelines[target] = pipeline.fit(_X_train, _y_train[target])
                y_pred = self.trained_pipelines[target].predict(_X_test)
                predictions_list.append(y_pred)
                print("test f1", target, f1_score(_y_test[target], y_pred))
                self.trained_pipelines[target] = pipeline.fit(X, y)
                y = _y_test["PCL_Strict3"]
            else:
                self.trained_pipelines[target] = pipeline.fit(X, y)
                predictions_list.append([self.trained_pipelines[target].predict(X)])
                y = y_train["PCL_Strict3"]

            if self.check_on_test_set:
                if target == "PCL_Strict3":
                    y_test = self.y_test[target].apply(lambda x: int(x))
                else:
                    y_test = X_train[target].apply(lambda x: int(x))
                if self.use_feature_engineering:
                    X_test = FeatureEngineering(self.X_test[self.features], target).engineer_features().values
                else:
                    X_test = self.X_test[self.features].values

                model = self.trained_pipelines[target]
                y_pred = model.predict(X_test)
                s_f = f1_score(self.y_test, y_pred)
                s_p = precision_score(self.y_test, y_pred)
                s_r = recall_score(self.y_test, y_pred)
                print(f"test f1 {target}", s_f)
                print(f"test recall {target}", s_r)
                print(f"test precision {target}", s_p)

        #pipe = Pipeline(steps=[
        #    ('scaling', StandardScaler()),
        #    ('sampling', SMOTE()),
        #    ('classifier', LogisticRegression(penalty='l1'))])
        #c = ((len(y) - sum(y)) / sum(y))

        if not self.use_and_func:
            c = 10
            pipe = Pipeline(steps=[('feature_selection',
                                    RFE(XGBClassifier(n_estimators=10, scale_pos_weight=c))),
                                   ('clf', XGBClassifier(scale_pos_weight=c))])
            X = predictions_list
            self.combined_model = pipe.fit(np.array(X).reshape(-1, len(predictions_list)), y)

    def predict(self, X_test):

        predictions_list = []
        for target in self.targets_list:
            if self.use_feature_engineering:
                X = FeatureEngineering(X_test[self.features], target).engineer_features().values
            else:
                X = X_test[self.features].values

            predictions_list.append([self.trained_pipelines[target].predict(X)])

        X = predictions_list
        if self.use_and_func:
            y_pred = 1
            for i in predictions_list:
                y_pred = (y_pred & np.array(i))
            y_pred = y_pred.reshape(-1,1)
        else:
            y_pred = self.combined_model.predict(np.array(X).reshape(-1, len(predictions_list)))

        return y_pred


def cv(X_train, y_train, features):

    kfold = LeaveOneOut() #StratifiedKFold(n_splits=3, shuffle=True)

    y_preds = []
    y_tests = []

    for train, test in kfold.split(X_train, y_train):
        model = TargetEnsembler(features)

        X_train_cv = pd.DataFrame(X_train.values[train], columns=X_train.columns)
        y_train_cv = pd.DataFrame(y_train.values[train], columns=["PCL_Strict3"])
        X_test_cv = pd.DataFrame(X_train.values[test], columns=X_train.columns)
        y_test_cv = pd.DataFrame(y_train.values[test], columns=["PCL_Strict3"])
        model.fit(X_train_cv, y_train_cv)
        y_preds.extend(model.predict(X_test_cv))
        y_tests.append(y_test_cv["PCL_Strict3"][0])

    s_f = f1_score(y_tests, y_preds)
    s_p = precision_score(y_tests, y_preds)
    s_r = recall_score(y_tests, y_preds)
    print("\tscores f1", (s_f))
    print("\tscores p", (s_p))
    print("\tscores r", (s_r))


def runner():

    targets_indexer = [
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
        ]
    for targets_index in targets_indexer:
        for counter, value in enumerate(targets_dict.keys()):
            targets_dict[value] = targets_index[counter]

        print(*[f"{target}_{targets_dict[target]}" for target in targets_dict])
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

        if check_on_test_set:

            print("real test set scores")
            model = TargetEnsembler(features, check_on_test_set=1, X_test=m.X_test_0, y_test=m.y_test_0)
            model.fit(m.X_train_0, m.y_train_0)

            y_pred = model.predict(m.X_test_0)
            s_f = f1_score(m.y_test_0, y_pred)
            s_p = precision_score(m.y_test_0, y_pred)
            s_r = recall_score(m.y_test_0, y_pred)
            print("test f1", s_f)
            print("test recall", s_r)
            print("test precision", s_p)


runner()
