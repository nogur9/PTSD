import xlsxwriter
from sklearn.model_selection import cross_val_score
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from itertools import combinations
import pandas as pd
from fancyimpute import KNN, IterativeImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
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

    def model_selection_by_grid_search(self, use_feature_engineering=1):

        if use_feature_engineering:
            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
        else:
            X = self.df[self.features]

        Y = self.df[self.target]
        c = ((len(Y) - sum(Y))/ sum(Y))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

        # _____________________________________________________________________________________________________________________


        # Construct some pipelines
        pipe_lr = Pipeline([('scl', StandardScaler()),
                            ('clf', LogisticRegression(random_state=42))])

        pipe = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        params = {'feature_selection': [
            RFE(XGBClassifier(n_estimators=100, reg_alpha=1, scale_pos_weight=c), n_features_to_select=n),
            SelectKBest(k=n), SelectFpr(alpha=1 / n), SelectFdr(alpha=1 / n)],
                  'sampling': [SMOTE(), BorderlineSMOTE()],
                  'sampling__k_neighbors': [5, 10],
                  'classifier': [RandomForestClassifier(n_estimators=100), XGBClassifier(),
                                 BalancedRandomForestClassifier()],
                  'classifier__n_estimators': [100, 200, 400, 600],
                  'classifier__max_depth': [2, 3, 5, 10]
                  }


        pipe_lr_pca = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])


        pipe_lr_pca = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])


        pipe_rf = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection',SelectKBest(k=n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])


        pipe_rf = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection',SelectFpr(alpha=1 / n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        pipe_svm_pca = Pipeline([('scl', StandardScaler()),
                                 ('feature_selection',SelectFpr(alpha=1 / n)),
                                  ('sampling', SMOTE()),
                                 ('clf', SVC(random_state=42))])

        pipe_rf_pca = Pipeline([('scl', StandardScaler()),
                                ('pca', PCA(n_components=2)),
                                ('clf', RandomForestClassifier(random_state=42))])

        pipe_lr_pca = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        pipe_lr_pca = Pipeline(steps=[
            ('feature_selection',
             RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        pipe_rf = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection', SelectKBest(k=n)),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        pipe_rf = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=1 / n)),
            ('sampling', SMOTE()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

        pipe_svm_pca = Pipeline([('scl', StandardScaler()),
                                 ('feature_selection', SelectFpr(alpha=1 / n)),
                                 ('clf', SVC(random_state=42))])
        pipe = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('sampling', SMOTE()),
            ('classifier', LogisticRegression(penalty='l1'))
        ])

        pipe = Pipeline(steps=[
            ('scl', StandardScaler()),
            ('feature_selection', SelectFpr(alpha=1 / n)),
            ('sampling', SMOTE()),
            ('classifier', LogisticRegression(penalty='l1'))
        ])

        params = {
            'sampling': [SMOTE(), BorderlineSMOTE()],
            'sampling__k_neighbors': [5, 10],
            'classifier__penalty': ['l1'],
            "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            #  "classifier__l1_ratio": np.arange(0.0, 1.0, 0.2)
        }
        # Set grid search params
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_range_fl = [1.0, 0.5, 0.1]

        grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                           'clf__C': param_range_fl,
                           'clf__solver': ['liblinear']}]

        grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                           'clf__min_samples_leaf': param_range,
                           'clf__max_depth': param_range,
                           'clf__min_samples_split': param_range[1:]}]

        grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
                            'clf__C': param_range}]

        # Construct grid searches
        jobs = -1

        gs_lr = GridSearchCV(estimator=pipe_lr,
                             param_grid=grid_params_lr,
                             scoring='accuracy',
                             cv=10)

        gs_lr_pca = GridSearchCV(estimator=pipe_lr_pca,
                                 param_grid=grid_params_lr,
                                 scoring='accuracy',
                                 cv=10)

        gs_rf = GridSearchCV(estimator=pipe_rf,
                             param_grid=grid_params_rf,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=jobs)

        gs_rf_pca = GridSearchCV(estimator=pipe_rf_pca,
                                 param_grid=grid_params_rf,
                                 scoring='accuracy',
                                 cv=10,
                                 n_jobs=jobs)

        gs_svm = GridSearchCV(estimator=pipe_svm,
                              param_grid=grid_params_svm,
                              scoring='accuracy',
                              cv=10,
                              n_jobs=jobs)

        gs_svm_pca = GridSearchCV(estimator=pipe_svm_pca,
                                  param_grid=grid_params_svm,
                                  scoring='accuracy',
                                  cv=10,
                                  n_jobs=jobs)

        # List of pipelines for ease of iteration
        grids = [gs_lr, gs_lr_pca, gs_rf, gs_rf_pca, gs_svm, gs_svm_pca]

        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'Logistic Regression', 1: 'Logistic Regression w/PCA',
                     2: 'Random Forest', 3: 'Random Forest w/PCA',
                     4: 'Support Vector Machine', 5: 'Support Vector Machine w/PCA'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_acc = 0.0
        best_clf = 0
        best_gs = ''
        for idx, gs in enumerate(grids):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            gs.fit(X_train, y_train)
            # Best params
            print('Best params: %s' % gs.best_params_)
            # Best training data accuracy
            print('Best training accuracy: %.3f' % gs.best_score_)
            # Predict on test data with best params
            y_pred = gs.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
            # Track best (highest test accuracy) model
            if accuracy_score(y_test, y_pred) > best_acc:
                best_acc = accuracy_score(y_test, y_pred)
                best_gs = gs
                best_clf = idx
        print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])


        if 0:
            # ____________________________________________________________________________________________________________________
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

        #_____________________________________________________________________________

            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
            Y = self.df[self.target]
            c = ((len(Y) - sum(Y)) / sum(Y))
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

            # ____________________________________________________________________________________

            X = FeatureEngineering(self.df[self.features], self.target).engineer_features()
            Y = self.df[self.target]
            c = ((len(Y) - sum(Y)) / sum(Y))
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

            pipe = Pipeline(steps=[
                ('feature_selection',
                 RFE(XGBClassifier(n_estimators=100, scale_pos_weight=c, reg_alpha=1), n_features_to_select=n)),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ])

            params = {'feature_selection': [
                RFE(XGBClassifier(n_estimators=100, reg_alpha=1, scale_pos_weight=c), n_features_to_select=n),
                SelectKBest(k=n), SelectFpr(alpha=1 / n), SelectFdr(alpha=1 / n)],
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

            #_____________________________________________________________________________

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









