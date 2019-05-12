from fancyimpute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from Model.model_object import Model
import pandas as pd
from sklearn.linear_model import LogisticRegression
features = ['trait1', 'trait2', 'lot1', 'PCL1', 'PCL2', 'phq2',
    'active_coping1', 'self_blame1',
    'HL_MAOA', 'HML_FKBP5', 'highschool_diploma', 'T1Acc1n', 'T1Acc1t',
    'q6.15_CONC_pcl1', 'q6.2_DREAM_pcl2'
    ]

class TargetEnsembler(object):

    rfe = 5
    n_estimators = 100
    cutoff = 50

    def __init__(self, features):
        print("self.n_estimators", self.n_estimators, "\nself.rfe", self.rfe, "\ncutoff", self.cutoff)
        self.features = features

    def fit(self, X_train, y_train):
        # intrusion
        X_intrusion = X_train[self.features].values
        y_intrusion = X_train["intrusion_cutoff"].apply(lambda x: int(x))
        self.pipe_intrusion = Pipeline(steps=[
           ('rfe',  RFE(XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=3), self.rfe)),
            ('classifier', XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=3))])
        self.pipe_intrusion.fit(X_intrusion, y_intrusion)
        scores = cross_val_score(self.pipe_intrusion, X_intrusion, y_intrusion, scoring='precision', cv=StratifiedKFold(5))
        print(f"intrusion {sum(scores)/5}")
        self.pipe_intrusion.fit(X_intrusion, y_intrusion)

        # avoidance
        X_avoidance = X_train[self.features].values
        y_avoidance = X_train["avoidance_cutoff"].apply(lambda x: int(x))
        self.pipe_avoidance = Pipeline(steps=[
            ('rfe', RFE(XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=6), self.rfe)),
            ('classifier', XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=6))])
        self.pipe_avoidance.fit(X_avoidance, y_avoidance)
        scores = cross_val_score(self.pipe_avoidance, X_avoidance, y_avoidance, scoring='precision', cv=StratifiedKFold(5))
        print(f"avoidance {sum(scores)/5}")
        self.pipe_avoidance.fit(X_avoidance, y_avoidance)

        # hypertension
        X_hypertension = X_train[self.features].values
        y_hypertention = X_train["hypertention_cutoff"].apply(lambda x: int(x))
        self.pipe_hypertension = Pipeline(steps=[
            ('rfe', RFE(XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=4), self.rfe)),
            ('classifier', XGBClassifier(n_estimators=self.n_estimators, reg_alpha=1, scale_pos_weight=4))])
        self.pipe_hypertension.fit(X_hypertension, y_hypertention)
        scores = cross_val_score(self.pipe_hypertension, X_hypertension, y_hypertention, scoring='precision', cv=StratifiedKFold(5))
        print(f"hypertension {sum(scores)/5}")
        self.pipe_hypertension.fit(X_hypertension, y_hypertention)

        # regression
        X_regression = X_train[self.features].values
        y_regression = X_train["PCL3"]
        self.pipe_regression = Pipeline(steps=[
            ('classifier', Ridge())])
        self.pipe_regression.fit(X_regression, y_regression)

        # target
        y_pred_hypertension = self.pipe_hypertension.predict(X_hypertension)
        y_pred_avoidance = self.pipe_avoidance.predict(X_avoidance)
        y_pred_intrusion = self.pipe_intrusion.predict(X_intrusion)
        y_pred_regression = self.pipe_regression.predict(X_regression) >= self.cutoff

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_regression & y_pred_regression)
        y_target = y_train

        acc = accuracy_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")



    def predict(self, X_test):
        ## combine three classifiers
        X_test_hypertension = X_test[self.features].values
        X_test_avoidance = X_test[self.features].values
        X_test_intrusion = X_test[self.features].values
        X_test_regression = X_test[self.features].values

        y_pred_hypertension = self.pipe_hypertension.predict(X_test_hypertension)
        y_pred_avoidance = self.pipe_avoidance.predict(X_test_avoidance)
        y_pred_intrusion = self.pipe_intrusion.predict(X_test_intrusion)
        y_pred_regression = self.pipe_regression.predict(X_test_regression) >= self.cutoff

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_regression & y_pred_regression)
        return y_pred

def cv(X_train, y_train):

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

def runner ():
    m = Model()
    X = m.df.drop("PCL_Strict3", axis=1)
    Y = m.df["PCL_Strict3"]

    cv(X, Y)
    features_inner = m.features + m.features_2
    model = TargetEnsembler(features_inner)
    model.fit(X, Y)

    y_pred = model.predict(m.X_test)
    s_f = f1_score(m.y_test, y_pred)
    s_p = precision_score(m.y_test, y_pred)
    s_r = recall_score(m.y_test, y_pred)
    print("test f1", s_f)
    print("test precision", s_p)
    print("test recall", s_r)




runner()