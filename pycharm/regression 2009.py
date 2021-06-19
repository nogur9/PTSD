import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from data_preparation import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
random_state = 854
score = roc_auc_score
df_2009, target_features_2009, features_2009 = extract_2009_data()


data_path_2009 = r'C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\questionnaires'


PCL_2009_1 = pd.read_csv(os.path.join(data_path_2009, "questionnaire_PCL1.csv"))

intrusion_features_2009 = ["q6.1_INTRU", "q6.2_DREAM", "q6.3_FLASH", "q6.4_UPSET", "q6.5_PHYS"]
avoidance_features_2009 = ["q6.6_AVTHT", "q6.7_AVSIT", "q6.8_AMNES", "q6.9_DISINT", "q6.10_DTACH",
             "q6.11_NUMB", "q6.12_FUTRE"]
hyper_features_2009 = ["q6.13_SLEEP", "q6.14_ANGER", "q6.15_CONC", "q6.16_HYPER", "q6.17_STRTL"]




PCL_2009_3 = pd.read_excel(os.path.join(data_path_2009, "questionnaire6PCL3.xlsx"))

PCL_2009_3 = df_2009.merge(PCL_2009_3[intrusion_features_2009 +avoidance_features_2009+hyper_features_2009+ ["ID"]], on="ID", how='outer')
df_2009['Intrusion_T4'] = PCL_2009_3[intrusion_features_2009].sum(axis=1)
df_2009['Avoidance_T4'] = PCL_2009_3[avoidance_features_2009].sum(axis=1)
df_2009['Hyper_T4'] = PCL_2009_3[hyper_features_2009].sum(axis=1)



df_2009 = df_2009.merge(PCL_2009_1[intrusion_features_2009 +avoidance_features_2009+hyper_features_2009+ ["ID"]], on="ID", how='outer')




PHQ_2009_1 = pd.read_csv(os.path.join(data_path_2009, "questionnaire_PHQ9.csv"))

phq_features_2009 = [f'T1q5.{i}' for i in range(1, 10)]


df_2009 = df_2009.merge(PHQ_2009_1[phq_features_2009 + ["ID"]], on="ID", how='outer')

df_2009 = df_2009[~df_2009[target_features_2009].isna().all(axis=1)]

# df_2009['Intrusion_T1'] = df_2009_1[intrusion_features_2009].sum(axis=1)
# df_2009['Avoidance_T1'] = df_2009_1[avoidance_features_2009].sum(axis=1)
# df_2009['Hyper_T1'] = df_2009_1[hyper_features_2009].sum(axis=1)
# features_2009.extend(['Intrusion_T1', 'Avoidance_T1', 'Hyper_T1'])
features_2009.extend(intrusion_features_2009 + avoidance_features_2009 + hyper_features_2009)
features_2009.extend(phq_features_2009)

target_feature = 'Intrusion_T4'
cv = StratifiedKFold(6, random_state=random_state, shuffle=True)

x_2009, y_2009 = df_2009[features_2009], df_2009[target_feature]
#x_2009 = x_2009.drop(['ABV'], axis=1)

for train, test in cv.split(x_2009, y_2009 > 8):
    x_train, y_train = x_2009.iloc[train], y_2009.iloc[train]
    x_test, y_test = x_2009.iloc[test], y_2009.iloc[test]

    lr_intrusion = CatBoostRegressor(verbose=0, random_state=random_state, l2_leaf_reg=100)

    # mice = IterativeImputer(max_iter=1000)
    # x_train = mice.fit_transform(x_train)
    # x_test = mice.transform(x_test)

    y_train_tr, y_test_tr = np.log(y_train+ 1)**4, np.log(y_test + 1)**2
    y_train_br, y_test_br = y_train > 8, y_test > 8

    lr_intrusion.fit(x_train, y_train_tr)

    y_pred_target = lr_intrusion.predict(x_test)
    y_pred_target_exp = np.exp(y_pred_target)**0.5
    y_pred_target_br = y_pred_target_exp > 8

    precision, recall, _ = precision_recall_curve(y_test_br.astype(int), y_pred_target_exp/y_pred_target_exp.max())
    auc_score = auc(recall, precision)
    print(f"pr_auc = {auc_score}")
#    print(f"roc_auc = {score(y_test_br.astype(int), y_pred_target_exp/y_pred_target_exp.max())}")

    for i in range(len(y_pred_target)):
        if y_pred_target_br[i] != y_test_br.astype(int).iloc[i]:
            pass
            #print(y_pred_target_br[i], y_test_br.astype(int).iloc[i])
            #print(df_2009['PCL3'].iloc[test].iloc[i])

