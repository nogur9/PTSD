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

targets = {
    'intrusion': 0,
    'avoidance': 0,
    'hypertension': 0,
    'depression': 0,
    'only_avoidance': 0,
    'PCL_Strict3': 1,
    'regression_cutoff_33': 0,
    'regression_cutoff_50': 1,
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

    def __init__(self, features):
        self.features = features

    def fit(self, X_train, y_train):

        # create list of targets

        # self.pipelines_list = []
        # self.preds = []
        # for i in targets :
        #  x. feature engineering (i)
        # y = df[i]
        # cv_scores  (x, y, pipeline_per_target[i])
        # model = pipeline_per_target[i].train(x, y)
        # pipelines_list.append(model)
        # preds.append(model.pred(x))

        # y = df[y]
        # combined_model = LogReg.train(preds, y)
        # print results....

        # def pred(X):
        #
        if intrusion:
            y_pred_intrusion = self.pipe_intrusion.predict(X_intrusion)
        else:
            y_pred_intrusion = 1

        if avoidance:
            y_pred_avoidance = self.pipe_avoidance.predict(X_avoidance)
        else:
            y_pred_avoidance = 1

        if hypertension:
            y_pred_hypertension = self.pipe_hypertension.predict(X_hypertension)
        else:
            y_pred_hypertension = 1

        if depression:
            y_pred_depression = self.pipe_depression.predict(X_depression)
        else:
            y_pred_depression = 1

        if only_avoidance:
            y_pred_only_avoidance = self.pipe_only_avoidance.predict(X_only_avoidance)
        else:
            y_pred_only_avoidance = 1

        if PCL_Strict3:
            y_pred_PCL_Strict3 = self.pipe_PCL_Strict3.predict(X_PCL_Strict3)
        else:
            y_pred_PCL_Strict3 = 1

        if regression_cutoff_33:
            y_pred_regression_cutoff_33 = self.pipe_regression_cutoff_33.predict(X_regression_cutoff_33)
        else:
            y_pred_regression_cutoff_33 = 1

        if regression_cutoff_50:
            y_pred_regression_cutoff_50 = self.pipe_regression_cutoff_50.predict(X_regression_cutoff_50)
        else:
            y_pred_regression_cutoff_50 = 1

        if tred_cutoff:
            y_pred_tred_cutoff = self.pipe_tred_cutoff.predict(X_tred_cutoff)
        else:
            y_pred_tred_cutoff = 1

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

        # combined
        y_pred_hypertension = self.pipe_hypertension.predict(X_hypertension)
        y_pred_avoidance = self.pipe_avoidance.predict(X_avoidance)
        y_pred_intrusion = self.pipe_intrusion.predict(X_intrusion)
        y_pred_regression = self.pipe_regression.predict(X_regression)

        X_train["y_pred_hypertension"] = y_pred_hypertension
        X_train["y_pred_avoidance"] = y_pred_avoidance
        X_train["y_pred_intrusion"] = y_pred_intrusion
        X_train["y_pred_regression"] = y_pred_regression
        preds = ["y_pred_hypertension", "y_pred_avoidance", "y_pred_intrusion", "y_pred_regression"]

        X_combined = X_train[['q6.11_NUMB_pcl2', 'q6.13_SLEEP_pcl1', 'intrusion_pcl2', 'phq2'] + preds].values
        y_combined = y_train
        self.pipe_combined = Pipeline(steps=[
            ('classifier', DecisionTreeClassifier())])
        scores = cross_val_score(self.pipe_combined, X_combined, y_combined, scoring='precision', cv=StratifiedKFold(5))
        print(f"hypertension {sum(scores)/5}")
        self.pipe_combined.fit(X_combined, y_combined)

    def predict(self, X_test):

        if intrusion:
            X_test_intrusion_cutoff = FeatureEngineering(X_test[self.features],
                                                         "intrusion_cutoff").engineer_features().values
            y_pred_intrusion = self.pipe_intrusion.predict(X_test_intrusion_cutoff)
        else:
            y_pred_intrusion = 1

        if avoidance:
            X_test_avoidance_cutoff = FeatureEngineering(X_test[self.features],
                                                         "avoidance_cutoff").engineer_features().values
            y_pred_avoidance = self.pipe_avoidance.predict(X_test_avoidance_cutoff)
        else:
            y_pred_avoidance = 1

        if hypertension:
            X_test_hypertention_cutoff = FeatureEngineering(X_test[self.features],
                                                            "hypertention_cutoff").engineer_features().values
            y_pred_hypertension = self.pipe_hypertension.predict(X_test_hypertention_cutoff)
        else:
            y_pred_hypertension = 1

        if depression:
            X_test_depression_cutoff = FeatureEngineering(X_test[self.features],
                                                          "depression_cutoff").engineer_features().values
            y_pred_depression = self.pipe_depression.predict(X_test_depression_cutoff)
        else:
            y_pred_depression = 1

        if only_avoidance:
            X_test_only_avoidance_cutoff = FeatureEngineering(X_test[self.features],
                                                              "only_avoidance_cutoff").engineer_features().values

            y_pred_only_avoidance = self.pipe_only_avoidance.predict(X_test_only_avoidance_cutoff)
        else:
            y_pred_only_avoidance = 1

        if PCL_Strict3:
            X_test_PCL_Strict3 = FeatureEngineering(X_test[self.features], "PCL_Strict3").engineer_features().values
            y_pred_PCL_Strict3 = self.pipe_PCL_Strict3.predict(X_test_PCL_Strict3)
        else:
            y_pred_PCL_Strict3 = 1

        if regression_cutoff_33:
            X_test_regression_cutoff_33 = FeatureEngineering(X_test[self.features],
                                                             "regression_cutoff_33").engineer_features().values
            y_pred_regression_cutoff_33 = self.pipe_regression_cutoff_33.predict(X_test_regression_cutoff_33)
        else:
            y_pred_regression_cutoff_33 = 1

        if regression_cutoff_50:
            X_test_regression_cutoff_50 = FeatureEngineering(X_test[self.features],
                                                             "regression_cutoff_50").engineer_features().values
            y_pred_regression_cutoff_50 = self.pipe_regression_cutoff_50.predict(X_test_regression_cutoff_50)
        else:
            y_pred_regression_cutoff_50 = 1

        if tred_cutoff:
            X_test_tred_cutoff = FeatureEngineering(X_test[self.features], "tred_cutoff").engineer_features().values
            y_pred_tred_cutoff = self.pipe_tred_cutoff.predict(X_test_tred_cutoff)
        else:
            y_pred_tred_cutoff = 1

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_depression &
                  y_pred_only_avoidance & y_pred_PCL_Strict3 & y_pred_regression_cutoff_33 &
                  y_pred_regression_cutoff_50 & y_pred_tred_cutoff)

        preds = ["y_pred_hypertension", "y_pred_avoidance", "y_pred_intrusion", "y_pred_regression"]

        X_combined = X_test[['q6.11_NUMB_pcl2', 'q6.13_SLEEP_pcl1', 'intrusion_pcl2', 'phq2'] + preds].values

        y_pred = self.pipe_combined.predict(X_combined)
        return y_pred


def cv(X_train, y_train):
    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    scores_f = []
    scores_p = []
    scores_r = []

    for train, test in kfold.split(X_train, y_train):
        model = TargetEnsembler(intrusion_features, avoidance_features, hypertension_features, regression_features)
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
    m = Model()
    X = m.df.drop("PCL_Strict3", axis=1)
    Y = m.df["PCL_Strict3"]

    cv(X, Y)

    model = TargetEnsembler(intrusion_features, avoidance_features, hypertension_features, regression_features)
    model.fit(X, Y)

    y_pred = model.predict(m.X_test)
    s_f = f1_score(m.y_test, y_pred)
    s_p = precision_score(m.y_test, y_pred)
    s_r = recall_score(m.y_test, y_pred)
    print("test f1", s_f)
    print("test recall", s_r)
    print("test precision", s_p)


runner()
