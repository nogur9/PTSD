from fancyimpute import IterativeImputer
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFpr, SelectKBest, SelectFdr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from Model.model_object import Model
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from feature_engineering.engineering import FeatureEngineering


intrusion = 0

avoidance = 0

hypertension = 1

depression = 1

only_avoidance = 0

PCL_Strict3 = 0

regression_cutoff_33 = 1

regression_cutoff_50 = 0

tred_cutoff = 0


class TargetEnsembler(object):

    def __init__(self, features):
        self.features = features

    def fit(self, X_train, y_train):

        # intrusion
        if intrusion:
            X_intrusion = FeatureEngineering(X_train[self.features], "intrusion_cutoff").engineer_features().values
            y_intrusion = X_train["intrusion_cutoff"].apply(lambda x: int(x))

            self.pipe_intrusion = Pipeline(steps=[
                ('feature_selection', SelectFpr(alpha=0.05)),
                ('sampling', BorderlineSMOTE(k_neighbors=10)),
                ('classifier', XGBClassifier(n_estimators=300, max_depth=5))])

            scores = cross_val_score(self.pipe_intrusion, X_intrusion, y_intrusion, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"intrusion {sum(scores)/5}")
            self.pipe_intrusion.fit(X_intrusion, y_intrusion)

        # avoidance
        if avoidance:
            X_avoidance = FeatureEngineering(X_train[self.features], "avoidance_cutoff").engineer_features().values
            y_avoidance = X_train["avoidance_cutoff"].apply(lambda x: int(x))

            self.pipe_avoidance = Pipeline(steps=[
                ('feature_selection',  RFE(estimator=XGBClassifier(scale_pos_weight=5.88, n_estimators=100),
                                           n_features_to_select=30)),
                ('classifier', RandomForestClassifier(n_estimators=500, max_depth=5))])

            scores = cross_val_score(self.pipe_avoidance, X_avoidance, y_avoidance, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"avoidance {sum(scores)/5}")
            self.pipe_avoidance.fit(X_avoidance, y_avoidance)

        # hypertension
        if hypertension:
            X_hypertension = FeatureEngineering(X_train[self.features], "hypertention_cutoff").engineer_features().values
            y_hypertention = X_train["hypertention_cutoff"].apply(lambda x: int(x))

            self.pipe_hypertension = Pipeline(steps=[
                ('feature_selection',  RFE(estimator=XGBClassifier(n_estimators=100, scale_pos_weight=3.51),
                                           n_features_to_select=30)),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3))])

            scores = cross_val_score(self.pipe_hypertension, X_hypertension, y_hypertention, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"hypertension {sum(scores)/5}")
            self.pipe_hypertension.fit(X_hypertension, y_hypertention)

        # depression
        if depression:
            X_depression = FeatureEngineering(X_train[self.features], "depression_cutoff").engineer_features().values
            y_depression = X_train["depression_cutoff"].apply(lambda x: int(x))

            self.pipe_depression = Pipeline(steps=[
                ('feature_selection', SelectFdr(alpha=0.033)),
                ('classifier', RandomForestClassifier(n_estimators=500, max_depth=10))])

            scores = cross_val_score(self.pipe_depression, X_depression, y_depression, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"depression {sum(scores)/5}")
            self.pipe_depression.fit(X_depression, y_depression)

        # only_avoidance
        if only_avoidance:
            X_only_avoidance = FeatureEngineering(X_train[self.features], "only_avoidance_cutoff").engineer_features().values
            y_only_avoidance = X_train["only_avoidance_cutoff"].apply(lambda x: int(x))

            self.pipe_only_avoidance = Pipeline(steps=[
                ('feature_selection', SelectFdr(alpha=0.033)),
                ('sampling', BorderlineSMOTE(k_neighbors=5)),
                ('classifier', RandomForestClassifier( n_estimators=100, max_depth=5))])

            scores = cross_val_score(self.pipe_only_avoidance, X_only_avoidance,
                                     y_only_avoidance, scoring='precision', cv=StratifiedKFold(5))
            print(f"only_avoidance {sum(scores)/5}")
            self.pipe_only_avoidance.fit(X_only_avoidance, y_only_avoidance)

        # pcl_strict3
        if PCL_Strict3:
            X_PCL_Strict3 = FeatureEngineering(X_train[self.features], "PCL_Strict3").engineer_features().values
            y_PCL_Strict3 = y_train["PCL_Strict3"].apply(lambda x: int(x))

            self.pipe_PCL_Strict3 = Pipeline(steps=[
                ('sampling', BorderlineSMOTE(k_neighbors=5)),
                ('classifier', LogisticRegression(C=100, penalty='l1'))])

            scores = cross_val_score(self.pipe_PCL_Strict3, X_PCL_Strict3,
                                     y_PCL_Strict3, scoring='precision', cv=StratifiedKFold(5))
            print(f"PCL_Strict3 {sum(scores)/5}")
            self.pipe_PCL_Strict3.fit(X_PCL_Strict3, y_PCL_Strict3)


        # cutoff_33
        if regression_cutoff_33:
            X_regression_cutoff_33 = FeatureEngineering(X_train[self.features],
                                                        "regression_cutoff_33").engineer_features().values
            y_regression_cutoff_33 = X_train["regression_cutoff_33"].apply(lambda x: int(x))

            self.pipe_regression_cutoff_33 = Pipeline(steps=[
                ('feature_selection', RFE(estimator=XGBClassifier(n_estimators=100, max_depth=3),
                                          n_features_to_select=30)),
                ('classifier', RandomForestClassifier(n_estimators=500, max_depth=3))])

            scores = cross_val_score(self.pipe_regression_cutoff_33, X_regression_cutoff_33,
                                     y_regression_cutoff_33, scoring='precision', cv=StratifiedKFold(5))
            print(f"regression_cutoff_33 {sum(scores)/5}")
            self.pipe_regression_cutoff_33.fit(X_regression_cutoff_33, y_regression_cutoff_33)

        # cutoff 50
        if regression_cutoff_50:
            X_regression_cutoff_50 = FeatureEngineering(X_train[self.features], "regression_cutoff_50").engineer_features().values
            y_regression_cutoff_50 = X_train["regression_cutoff_50"].apply(lambda x: int(x))

            self.pipe_regression_cutoff_50 = Pipeline(steps=[
                ('feature_selection', SelectFdr(alpha=0.1)),
                ('sampling', BorderlineSMOTE(k_neighbors=10)),
                ('classifier', XGBClassifier(max_depth=3, n_estimators=500))])

            scores = cross_val_score(self.pipe_regression_cutoff_50, X_regression_cutoff_50,
                                     y_regression_cutoff_50, scoring='precision', cv=StratifiedKFold(5))
            print(f"regression_cutoff_50 {sum(scores)/5}")
            self.pipe_regression_cutoff_50.fit(X_regression_cutoff_50, y_regression_cutoff_50)

        # tred_cutoff
        if tred_cutoff:
            X_tred_cutoff = FeatureEngineering(X_train[self.features], "tred_cutoff").engineer_features().values
            y_tred_cutoff = X_train["tred_cutoff"].apply(lambda x: int(x))

            self.pipe_tred_cutoff = Pipeline(steps=[
                ('feature_selection', SelectFdr(alpha=0.1)),
                ('classifier', RandomForestClassifier(n_estimators=300, max_depth=10))])

            scores = cross_val_score(self.pipe_tred_cutoff, X_tred_cutoff, y_tred_cutoff, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"tred_cutoff {sum(scores)/5}")
            self.pipe_tred_cutoff.fit(X_tred_cutoff, y_tred_cutoff)

        # target
        if intrusion:
            y_pred_intrusion = self.pipe_intrusion.predict(X_intrusion)
        else:
            y_pred_intrusion = 1

        if avoidance:
            y_pred_avoidance = self.pipe_avoidance.predict(X_avoidance)
        else: y_pred_avoidance = 1

        if hypertension:
            y_pred_hypertension = self.pipe_hypertension.predict(X_hypertension)
        else: y_pred_hypertension = 1

        if depression:
            y_pred_depression = self.pipe_depression.predict(X_depression)
        else: y_pred_depression = 1

        if only_avoidance:
            y_pred_only_avoidance = self.pipe_only_avoidance.predict(X_only_avoidance)
        else: y_pred_only_avoidance = 1

        if PCL_Strict3:
            y_pred_PCL_Strict3 = self.pipe_PCL_Strict3.predict(X_PCL_Strict3)
        else: y_pred_PCL_Strict3 = 1

        if regression_cutoff_33:
            y_pred_regression_cutoff_33 = self.pipe_regression_cutoff_33.predict(X_regression_cutoff_33)
        else: y_pred_regression_cutoff_33 = 1

        if regression_cutoff_50:
            y_pred_regression_cutoff_50 = self.pipe_regression_cutoff_50.predict(X_regression_cutoff_50)
        else: y_pred_regression_cutoff_50 = 1

        if tred_cutoff:
            y_pred_tred_cutoff = self.pipe_tred_cutoff.predict(X_tred_cutoff)
        else: y_pred_tred_cutoff = 1


        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_depression &
                  y_pred_only_avoidance & y_pred_PCL_Strict3 & y_pred_regression_cutoff_33 &
                  y_pred_regression_cutoff_50 & y_pred_tred_cutoff)
        y_target = y_train

        acc = accuracy_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        print("training scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")

    def predict(self, X_test):


        if intrusion:
            X_test_intrusion_cutoff = FeatureEngineering(X_test[self.features],
                                                         "intrusion_cutoff").engineer_features().values
            y_pred_intrusion = self.pipe_intrusion.predict(X_test_intrusion_cutoff)
        else: y_pred_intrusion = 1

        if avoidance:
            X_test_avoidance_cutoff = FeatureEngineering(X_test[self.features],
                                                         "avoidance_cutoff").engineer_features().values
            y_pred_avoidance = self.pipe_avoidance.predict(X_test_avoidance_cutoff)
        else: y_pred_avoidance = 1

        if hypertension:
            X_test_hypertention_cutoff = FeatureEngineering(X_test[self.features],
                                                            "hypertention_cutoff").engineer_features().values
            y_pred_hypertension = self.pipe_hypertension.predict(X_test_hypertention_cutoff)
        else: y_pred_hypertension = 1

        if depression:
            X_test_depression_cutoff = FeatureEngineering(X_test[self.features],
                                                          "depression_cutoff").engineer_features().values
            y_pred_depression = self.pipe_depression.predict(X_test_depression_cutoff)
        else: y_pred_depression = 1

        if only_avoidance:
            X_test_only_avoidance_cutoff = FeatureEngineering(X_test[self.features],
                                                              "only_avoidance_cutoff").engineer_features().values

            y_pred_only_avoidance = self.pipe_only_avoidance.predict(X_test_only_avoidance_cutoff)
        else: y_pred_only_avoidance = 1

        if PCL_Strict3:
            X_test_PCL_Strict3 = FeatureEngineering(X_test[self.features], "PCL_Strict3").engineer_features().values
            y_pred_PCL_Strict3 = self.pipe_PCL_Strict3.predict(X_test_PCL_Strict3)
        else: y_pred_PCL_Strict3 = 1

        if regression_cutoff_33:
            X_test_regression_cutoff_33 = FeatureEngineering(X_test[self.features],
                                                             "regression_cutoff_33").engineer_features().values
            y_pred_regression_cutoff_33 = self.pipe_regression_cutoff_33.predict(X_test_regression_cutoff_33)
        else: y_pred_regression_cutoff_33 =1

        if regression_cutoff_50:
            X_test_regression_cutoff_50 = FeatureEngineering(X_test[self.features],
                                                             "regression_cutoff_50").engineer_features().values
            y_pred_regression_cutoff_50 = self.pipe_regression_cutoff_50.predict(X_test_regression_cutoff_50)
        else: y_pred_regression_cutoff_50 = 1

        if tred_cutoff:
            X_test_tred_cutoff = FeatureEngineering(X_test[self.features], "tred_cutoff").engineer_features().values
            y_pred_tred_cutoff = self.pipe_tred_cutoff.predict(X_test_tred_cutoff)
        else: y_pred_tred_cutoff = 1

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_depression &
                  y_pred_only_avoidance & y_pred_PCL_Strict3 & y_pred_regression_cutoff_33 &
                  y_pred_regression_cutoff_50 & y_pred_tred_cutoff)

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
    print(f" intrusion = {intrusion} \n avoidance = {avoidance} \n hypertension = {hypertension} \n"
          f" depression = {depression} \n only_avoidance = {only_avoidance} \n PCL_Strict3 = {PCL_Strict3} \n"
          f" regression_cutoff_33 = {regression_cutoff_33} \n regression_cutoff_50 = {regression_cutoff_50} \n"
          f" tred_cutoff = {tred_cutoff}")

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
