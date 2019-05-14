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

def cv(X_train, y_train, features_inner):

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    scores_f = []
    scores_p = []
    scores_r = []

    for train, test in kfold.split(X_train, y_train):

        model = XGBClassifier()
        X_train_cv = pd.DataFrame(X_train.values[train], columns=X_train.columns)
        y_train_cv = pd.DataFrame(y_train.values[train], columns=["tred_cutoff"])
        X_test_cv = pd.DataFrame(X_train.values[test], columns=X_train.columns)
        y_test_cv = pd.DataFrame(y_train.values[test], columns=["tred_cutoff"])
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
    X = m.df.drop("tred_cutoff", axis=1)
    Y = m.df["tred_cutoff"]
    features_inner = m.features + m.features_2
    cv(X, Y, features_inner)

    model = XGBClassifier()
    model.fit(X, Y)

    y_pred = model.predict(m.X_test)
    s_f = f1_score(m.y_test, y_pred)
    s_p = precision_score(m.y_test, y_pred)
    s_r = recall_score(m.y_test, y_pred)
    print("test f1", s_f)
    print("test precision", s_p)
    print("test recall", s_r)




runner()