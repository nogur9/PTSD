import xlsxwriter
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import RFE, SelectKBest, SelectFdr, SelectFpr
import string
import datawig
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression
from feature_engineering.engineering import FeatureEngineering
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from itertools import combinations
import pandas as pd
from fancyimpute import KNN, IterativeImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class EDAMultiFeatureBackend:

    def __init__(self, df, features, target, plot_path=r"Visualization/plots"):

        self.file_path = os.path.join("feature_summery", target)
        self.target = target
        df = df[~df[target].isna()]
        self.df = df
        self.features = features
        # add the missing number to the data
            # if impute_missing_values is an lambda function of imputation method fill the values with it
        self.plot_path = os.path.join(plot_path, 'knn_imputation', target)
        self.df = self.impute(self.df)
        self.inteactions_file_name = "sum interaction EDA multiple Features outer merge Mice imputation.xlsx"



    def impute(self, df):

        mice = IterativeImputer()
        return pd.DataFrame(mice.fit_transform(df), columns=df.columns)

    def model_selection_by_grid_search(self, use_feature_engineering=0):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


        pipe_rf = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier())
        ])


        params_rf = {
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  #'sampling__k_neighbors': [5, 10, 15],
                  #'classifier': [RandomForestClassifier()],
                  'classifier__n_estimators': [100, 250, 400, 700],
                  'classifier__max_depth': [2, 3, 6, 10],
                  'classifier__max_features': [1.0, 0.8, 0.5, 'auto'],
                  'classifier__min_samples_split': [1.0, 0.9, 0.8, 0.5]
                  }


        pipe_bbc = Pipeline(steps=[
            ('classifier', BalancedBaggingClassifier())
        ])


        params_bbc = {
                  'classifier': [BalancedRandomForestClassifier(), BalancedBaggingClassifier()],
                  'classifier__n_estimators': [100, 250, 400, 700],
                  'classifier__max_features': [1.0, 0.9, 0.8,0.5],
                  }

        pipe_xgb = Pipeline(steps=[
            ('classifier', XGBClassifier())
        ])

        params_xgb = {
                  'classifier': [XGBClassifier()],
                  'classifier__n_estimators': [100, 250, 400, 700],
                  'classifier__max_depth': [2, 3, 6, 10],
                  'classifier__learning_rate': [0.1, 0.05, 0.25],
                  'classifier__gamma': [0, 0.5, 1],
                  #'classifier__min_child_weight': [1, 0.5, 2],
                  'classifier__scale_pos_weight': [c, 2*c, 0.5*c],
                  #'classifier__reg_alpha': [0, 1, 0.5],
                  #'classifier__reg_lambda': [0, 1, 0.5]
                  }

        pipe_lr = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE()),
            ('classifier', LogisticRegression(solver='warn'))
        ])

        params_lr = {
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  #'sampling__k_neighbors': [5, 10, 15],
                  'classifier__penalty': ['l1', 'l2'],
                  "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  }

        pipe_nn = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE()),
            ('classifier', MLPClassifier())
        ])


        params_nn = {
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  #'sampling__k_neighbors': [5, 10, 15],
                  'classifier__hidden_layer_sizes': [(100,), (10,), (10, 10), (10, 5), (100, 10), (10, 10, 10)],
                  'classifier__alpha': [0.0001, 0.001, 0.00001],
                  'classifier__learning_rate': ['constant', 'invscaling', 'adaptive']
                  }
        loo = StratifiedKFold(5)

        gs_rf = GridSearchCV(estimator=pipe_rf,
                             param_grid=params_rf,
                             scoring='f1',
                             cv=loo)

        gs_bbc = GridSearchCV(estimator=pipe_bbc,
                             param_grid=params_bbc,
                             scoring='f1',
                             cv=loo)

        gs_xgb = GridSearchCV(estimator=pipe_xgb,
                             param_grid=params_xgb,
                             scoring='f1',
                             cv=loo)

        gs_lr = GridSearchCV(estimator=pipe_lr,
                             param_grid=params_lr,
                             scoring='f1',
                             cv=loo)


        gs_nn = GridSearchCV(estimator=pipe_nn,
                             param_grid=params_nn,
                             scoring='f1',
                             cv=loo)

        # List of pipelines for ease of iteration
        grids = [gs_nn, gs_bbc, gs_xgb,
                 gs_lr, gs_rf]

        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'nn', 1: 'bbc', 2: 'xgb',
                     3: 'lr', 4: 'rf'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, gs in enumerate(grids):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            gs.fit(X_train, y_train)
            # Best params
            print('Best params: %s' % gs.best_params_)
            # Best training data accuracy
            print('Best training score: %.3f' % gs.best_score_)
            # Predict on test data with best params
            y_pred = gs.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])



    def model_selection_by_grid_search_loo(self, use_feature_engineering=0):
        print(f"use_feature_engineering = {use_feature_engineering}")
        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50

        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100,  max_depth=7))
        ])


        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ])



        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ])
        pipe_13 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFdr(alpha=0.01)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_14 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',SelectFdr(alpha=0.5)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_15 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFdr(alpha=0.01)),
            ('sampling', BorderlineSMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_16 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFdr(alpha=0.5)),
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_17 = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('classifier', BalancedBaggingClassifier(n_estimators=200))
        ])

        pipe_18 = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=c))
        ])

        pipe_19 = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_20 = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_21 = Pipeline(steps=[
            ('feature_selection', SelectKBest(k=10)),
            ('classifier', BalancedBaggingClassifier(n_estimators=400))
        ])

        pipe_22 = Pipeline(steps=[
            ('feature_selection',SelectKBest(k=10)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_23 = Pipeline(steps=[
            ('feature_selection', SelectKBest(k=25)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_24 = Pipeline(steps=[
            ('feature_selection', SelectKBest(k=25)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7, scale_pos_weight=c))
        ])

        pipe_25 = Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('classifier', BalancedBaggingClassifier(n_estimators=400))
        ])

        pipe_26 = Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_27 = Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=3, scale_pos_weight=c))
        ])

        pipe_28 = Pipeline(steps=[
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7, scale_pos_weight=c))
        ])
        pipe_29 = Pipeline(steps=[
            ('feature_selection', SelectFdr(alpha=0.01)),
            ('classifier', BalancedBaggingClassifier(n_estimators=400))
        ])

        pipe_30 = Pipeline(steps=[
            ('feature_selection',SelectFdr(alpha=0.5)),
            ('classifier', BalancedRandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_31 = Pipeline(steps=[
            ('feature_selection', SelectFdr(alpha=0.01)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=3, scale_pos_weight=c))
        ])

        pipe_32 = Pipeline(steps=[
            ('feature_selection', SelectFdr(alpha=0.5)),
            ('classifier', XGBClassifier(n_estimators=400, max_depth=7, scale_pos_weight=c))
        ])


        pipe_33 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', BorderlineSMOTE(k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=2))
        ])

        pipe_34 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=30))
        ])

        pipe_35 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', BorderlineSMOTE(k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=30))
        ])

        pipe_36 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=0.1))
        ])


        pipe_37 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', penalty='l2', C=0.1))
        ])

        pipe_38 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=30))
        ])

        pipe_39 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', penalty='l1', C=30))
        ])

        pipe_40 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', penalty='l2', C=0.1))
        ])


        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12,
                 pipe_13, pipe_14, pipe_15, pipe_16, pipe_17, pipe_18, pipe_19, pipe_20, pipe_21, pipe_22, pipe_23,
                 pipe_24, pipe_25, pipe_26, pipe_27, pipe_28, pipe_29, pipe_30, pipe_31, pipe_32, pipe_33, pipe_34,
                 pipe_35, pipe_36, pipe_37, pipe_38, pipe_39, pipe_40]
        #
        # pipes = [pipe_4,  pipe_8, pipe_12, pipe_16, pipe_20,
        #          pipe_24,  pipe_28, pipe_32,  pipe_36,  pipe_40]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12',
                     12: 'pipe_13', 13: 'pipe_14', 14:'pipe_15',
                     15: 'pipe_16', 16: 'pipe_17', 17: 'pipe_18',
                     18: 'pipe_19', 19: 'pipe_20', 20: 'pipe_21',
                     21: 'pipe_22', 22: 'pipe_23', 23: 'pipe_24',
                     24: 'pipe_25', 25: 'pipe_24', 26: 'pipe_27',
                     27: 'pipe_28', 28: 'pipe_29', 29: 'pipe_30',
                     30: 'pipe_31', 31: 'pipe_32', 32: 'pipe_33',
                     33: 'pipe_34', 34: 'pipe_35', 35: 'pipe_36',
                     36: 'pipe_37', 37: 'pipe_38', 38: 'pipe_39',
                     39: 'pipe_40'}

        # grid_dict = {0: 'pipe_4', 1: 'pipe_8', 2: 'pipe_12',
        #              3: 'pipe_16', 4: 'pipe_20', 5: 'pipe_24',
        #              6: 'pipe_28', 7: 'pipe_32', 8: 'pipe_36',
        #              9: 'pipe_40'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])




    def model_selection_by_grid_search_loo_diagnosis(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=300, penalty='l1'))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=600, penalty='l1'))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=10, penalty='l1'))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=3, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=1, penalty='l1'))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=3000, penalty='l1'))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('sampling', SMOTE(k_neighbors=7, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=30, penalty='l1'))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.8)),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=10, penalty='l1'))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.9)),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy=0.9)),
            ('classifier', LogisticRegression(solver='warn', C=10, penalty='l1'))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', SMOTE(k_neighbors=7, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=100, penalty='l1'))
        ])

        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(k_neighbors=3, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l1'))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(k_neighbors=3, sampling_strategy=0.9)),
            ('classifier', LogisticRegression(solver='warn', C=5000, penalty='l1'))
        ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('pca', PCA()),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=1, penalty='l1'))
        ])

        pipe_13 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.05)),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=10, penalty='l1'))
        ])

        pipe_14 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.2)),
            ('sampling', SMOTE(k_neighbors=5, sampling_strategy='auto')),
            ('classifier', LogisticRegression(solver='warn', C=30, penalty='l1'))
        ])
        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12, pipe_13, pipe_14]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12',
                     12: 'pipe_13', 13: 'pipe_14'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])


    def model_selection_by_grid_search_loo_tred(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.8)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=20)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=20)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.8)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.8)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=20)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=7)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])


        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])





    def model_selection_by_grid_search_loo_only_avoidance(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.005)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.05)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])


        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=20)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=7)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.01)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=7)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=20)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.005)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.005)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=15)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])



    def model_selection_by_grid_search_loo_hypertension(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.7)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])


        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=3)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy=0.9, k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.01, penalty='l2'))
            ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.5, penalty='l2'))
            ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.5)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.05, penalty='l2'))
            ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.001, penalty='l2'))
            ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.3)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.7)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', LogisticRegression(solver='warn', C=0.1, penalty='l2'))
            ])

        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])


    def model_selection_by_grid_search_loo_regression_cutoff_33(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.2)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.05)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])


        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=3)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=600, max_depth=7))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=10))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7,  min_samples_leaf=2))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=7)),
            ('classifier', RandomForestClassifier(n_estimators=600, max_depth=10,  min_samples_leaf=2))
        ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.1)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=5,  min_samples_leaf=2))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.05)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=3)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=5))
        ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=0.2)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=7)),
            ('classifier', RandomForestClassifier(n_estimators=600, max_depth=5))
        ])

        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])


    def model_selection_by_grid_search_loo_depression(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=15)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=7)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=15)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=7))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=500, max_depth=7))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=10))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=5))
        ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=2))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=2))
        ])

        pipe_12 =Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=2))
        ])


    def model_selection_by_grid_search_loo_regression_cutoff_50(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=35)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=3)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=7))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=11))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=7))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy=0.9, k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=11))
        ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=2))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=20)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2))
        ])

        pipe_12 =Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy=0.9, k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=2))
        ])


    def model_selection_by_grid_search_loo_intrusion(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=35)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=15)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=2))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=15)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=3))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=3))
        ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=3))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=4))
        ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2))
        ])

        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])


    def model_selection_by_grid_search_loo_avoidance(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
        # regression_cutoff_50
        pipe_1 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_2 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=15)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=3))
        ])

        pipe_3 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=3))
        ])

        pipe_4 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=7))
        ])

        pipe_5 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=11))
        ])

        pipe_6 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5))
        ])

        pipe_7 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=25)),
            ('sampling', SMOTE(sampling_strategy='auto', k_neighbors=5)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=7))
        ])

        pipe_8 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(k=10)),
            ('sampling', BorderlineSMOTE(sampling_strategy=0.9, k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=400, max_depth=7))
        ])

        pipe_9 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=3))
        ])


        pipe_10 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=3))
        ])

        pipe_11 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=4))
        ])

        pipe_12 = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=25)),
            ('sampling', BorderlineSMOTE(sampling_strategy='auto', k_neighbors=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2))
        ])

        pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8]#, pipe_9, pipe_10, pipe_11, pipe_12]
        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'pipe_1', 1: 'pipe_2', 2: 'pipe_3',
                     3: 'pipe_4', 4: 'pipe_5', 5: 'pipe_6',
                     6: 'pipe_7', 7: 'pipe_8', 8: 'pipe_9',
                     9: 'pipe_10', 10: 'pipe_11', 11: 'pipe_12'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_clf = 0
        best_f1 = 0.0
        for idx, pipe in enumerate(pipes):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            y_pred  = cross_val_predict(pipe, X_train, y_train, cv=LeaveOneOut())
            # Best params
            # Best training data accuracy
            print('training score: %.3f' % f1_score(y_train, y_pred))
            # Predict on test data with best params
            pipe = pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if f1_score(y_test, y_pred) > best_f1:
                best_f1 = f1_score(y_test, y_pred)
                best_clf = idx
        print('\nClassifier with best test set score: %s' % grid_dict[best_clf])


    def model_checking(self, n, scoring='f1'):
        X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

        pipe = Pipeline(steps=[
            ('feature_selection', RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        params = {'feature_selection':[RFE(XGBClassifier(n_estimators=100, reg_alpha=1, scale_pos_weight=c),  n_features_to_select=n),
                                       SelectKBest(k=n), SelectFpr(alpha=1/n), SelectFdr(alpha=1/n)],
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  'sampling__k_neighbors': [5, 10],
                  'classifier': [RandomForestClassifier(n_estimators=100), XGBClassifier(),
                                 BalancedRandomForestClassifier()],
                  'classifier__n_estimators': [100, 200, 400,600],
                  'classifier__max_depth': [2, 3, 5, 10]
                  }
        clf = GridSearchCV(pipe, params, cv=StratifiedKFold(5), scoring=scoring)

        clf.fit(X_train, y_train)
        print("clf.best_params_", clf.best_params_)
        print(f"best {scoring} score", clf.best_score_)

        y_pred = clf.best_estimator_.predict(X_test.values)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")

    def logistic_regression_grid_search(self, scoring = 'f1'):

        X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

        pipe = Pipeline(steps=[
            ('sampling', SMOTE()),
            ('classifier', LogisticRegression())
        ])

        params = {
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  'sampling__k_neighbors': [5, 10],
                  'classifier__penalty': ['l1'],
                  "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                #  "classifier__l1_ratio": np.arange(0.0, 1.0, 0.2)
                  }
        clf = GridSearchCV(pipe, params, cv=StratifiedKFold(5), scoring=scoring)

        clf.fit(X_train, y_train)
        print("clf.best_params_", clf.best_params_)
        print(f"best {scoring} score", clf.best_score_)

        y_pred = clf.best_estimator_.predict(X_test.values)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")


    def model_checking_without_resampling(self, n, scoring='f1'):
        X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

        pipe = Pipeline(steps=[
            ('feature_selection', RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        params = {'feature_selection':[RFE(XGBClassifier(n_estimators=100, reg_alpha=1, scale_pos_weight=c), n_features_to_select=n),
                                       SelectKBest(k=n), SelectFpr(alpha=1/n), SelectFdr(alpha=1/n)],
                  'classifier': [RandomForestClassifier(n_estimators=100), XGBClassifier(scale_pos_weight=c),
                                 BalancedRandomForestClassifier()],
                  'classifier__n_estimators': [100, 300, 500],
                  'classifier__max_depth': [10, 3, 5]
                  }
        clf = RandomizedSearchCV(pipe, params, cv=StratifiedKFold(5), scoring=scoring)

        clf.fit(X_train, y_train)
        print("clf.best_params_", clf.best_params_)
        print(f"best {scoring} score", clf.best_score_)

        y_pred = clf.best_estimator_.predict(X_test.values)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")

    def illigal_genralization_checking(self, X_test, y_test):

        X = self.df[self.features]
        X_test = X_test[self.features]
        Y = self.df[self.target]
        pipe = Pipeline(steps=[('classifier', XGBClassifier(n_estimators=1000, scale_pos_weight=3, reg_alpha=1))])
        y_test = y_test["intrusion_cutoff"].apply(lambda x: int(x))
        scores = cross_val_score(pipe, X, Y, scoring='precision', cv=StratifiedKFold(5))
        print(self.features)
        print("cross vl scores")
        print(sum(scores)/5)
        pipe.fit(X, Y.values)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")



    def three_models_combined(self, intrusion_features, avoidance_features, hypertension_features, regression_features,
                              depression_features):

        self.df = self.df[~self.df['intrusion_cutoff'].isna()]
        self.df = self.df[~self.df['avoidance_cutoff'].isna()]
        self.df = self.df[~self.df['hypertention_cutoff'].isna()]
        self.df = self.df[~self.df['PCL3'].isna()]

        print("self.df.shape", self.df.shape)
        X = self.df
        Y = self.df[self.target]# strict
        all_Y = [self.target, "intrusion_cutoff", "avoidance_cutoff", "hypertention_cutoff", "PCL3","depression_cutoff", "only_avoidance_cutoff"]

        X_train, X_test, y_train, y_test = train_test_split(X, self.df[all_Y], test_size=0.25, random_state=8526566, stratify=Y)

        # intrusion
        X_intrusion = X_train[intrusion_features].values
        y_intrusion = y_train["intrusion_cutoff"].apply(lambda x: int(x))
        pipe_intrusion = Pipeline(steps=[
            ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
            ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=3))])
        scores = cross_val_score(pipe_intrusion, X_intrusion, y_intrusion, scoring='precision', cv=StratifiedKFold(5))
        print(f"intrusion {sum(scores)/5}")
        pipe_intrusion.fit(X_intrusion, y_intrusion)

        # avoidance
        X_avoidance = X_train[avoidance_features].values
        y_avoidance = y_train["avoidance_cutoff"].apply(lambda x: int(x))
        pipe_avoidance = Pipeline(steps=[
            ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
            ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=6))])
        scores = cross_val_score(pipe_avoidance, X_avoidance, y_avoidance, scoring='precision', cv=StratifiedKFold(5))
        print(f"avoidance {sum(scores)/5}")
        pipe_avoidance.fit(X_avoidance, y_avoidance)

        # hypertension
        X_hypertension = X_train[hypertension_features].values
        y_hypertention = y_train["hypertention_cutoff"].apply(lambda x: int(x))
        pipe_hypertension = Pipeline(steps=[
            ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
            ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=4))])
        scores = cross_val_score(pipe_hypertension, X_hypertension, y_hypertention, scoring='precision', cv=StratifiedKFold(5))
        print(f"hypertension {sum(scores)/5}")
        pipe_hypertension.fit(X_hypertension, y_hypertention)

        # depression
        X_depression = X_train[depression_features].values
        y_depression = y_train["depression_cutoff"].apply(lambda x: int(x))
        pipe_depression = Pipeline(steps=[
            ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
        scores = cross_val_score(pipe_depression, X_depression, y_depression, scoring='precision', cv=StratifiedKFold(5))
        print(f"depression {sum(scores)/5}")
        pipe_depression.fit(X_depression, y_depression)

        # only_avoidance
        X_only_avoidance = X_train[avoidance_features].values
        y_only_avoidance = y_train["only_avoidance_cutoff"].apply(lambda x: int(x))
        pipe_only_avoidance = Pipeline(steps=[
            ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
        scores = cross_val_score(pipe_only_avoidance, X_only_avoidance, y_only_avoidance, scoring='precision', cv=StratifiedKFold(5))
        print(f"only_avoidance {sum(scores)/5}")
        pipe_only_avoidance.fit(X_only_avoidance, y_only_avoidance)

        # regression
        X_regression = X_train[regression_features].values
        y_regression = y_train["PCL3"]
        pipe_regression = Pipeline(steps=[
            ('classifier', RandomForestRegressor(n_estimators=500))])
        scores = cross_val_score(pipe_regression, X_regression, y_regression)
        print(f"regression {sum(scores)/5}")
        pipe_regression.fit(X_regression, y_regression)


        ## combine three classifiers
        X_test_hypertension = X_test[hypertension_features].values
        X_test_avoidance = X_test[avoidance_features].values
        X_test_intrusion = X_test[intrusion_features].values
        X_test_regression = X_test[regression_features].values
        X_test_depression = X_test[depression_features].values
        X_test_only_avoidance = X_test[avoidance_features].values

        y_pred_hypertension = pipe_hypertension.predict(X_test_hypertension)
        y_pred_avoidance = pipe_avoidance.predict(X_test_avoidance)
        y_pred_intrusion = pipe_intrusion.predict(X_test_intrusion)
        y_pred_regression = pipe_regression.predict(X_test_regression) >= 35
        y_pred_depression = pipe_depression.predict(X_test_depression)
        y_pred_only_avoidance = pipe_only_avoidance.predict(X_test_only_avoidance)

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_regression)

        y_target = y_test["PCL_Strict3"].apply(lambda x: int(x))

        acc = accuracy_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")



    def regression_model(self):

        X = self.df[self.features]
        y_reg = self.df[self.target]

        pipeline = Pipeline(steps=[
                ('classifier', RandomForestRegressor(n_estimators=500))])


        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.25)


        scores = cross_val_score(pipeline, X_train.values, y_train)
        print(pipeline)
        print("cross val scores")
        print(sum(scores)/5)
        pipeline.fit(X_train.values, y_train.values)
        y_pred = pipeline.predict(X_test.values) >= 35
        y_test = y_test >= 35
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")









