from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engineering.PCL_calculator import PCL_calculator

features = ["age", "highschool_diploma", "dyslexia", "ADHD", "T1Acc1t", "T1Acc1n", "T1bias", "phq1", "lot1",
            "trait1",
            "state1", "PCL1", "PCL_Broad1", "PCL_Strict1", "phq2", "lot2", "trait2", "state2", "PCL2", "PCL_Broad2",
            "PCL_Strict2", "cd_risc1", "active_coping1", "planning1", "positive_reframing1", "acceptance1",
            "humor1",
            "religion1", "emotional_support1", "instrumental_support1", "self_distraction1", "denial1",
            "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1", "active_coping2", "planning2",
            "positive_reframing2", "acceptance2", "humor2", "religion2", "emotional_support2",
            "instrumental_support2",
            "self_distraction2", "denial2", "venting2", "substance_use2", "behavioral_disengagement2",
            "self_blame2",
            "trauma_history8_1", "HML_5HTT", "HL_MAOA", "HML_NPY", "COMT_Hap1_recode",
            "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5"]

features_2 = ['q6.1_INTRU_pcl1', 'q6.2_DREAM_pcl1',
              'q6.3_FLASH_pcl1', 'q6.4_UPSET_pcl1',
              'q6.5_PHYS_pcl1', 'q6.6_AVTHT_pcl1', 'q6.7_AVSIT_pcl1', 'q6.8_AMNES_pcl1', 'q6.9_DISINT_pcl1',
              'q6.10_DTACH_pcl1', 'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl1', 'q6.13_SLEEP_pcl1',
              'q6.14_ANGER_pcl1', 'q6.15_CONC_pcl1', 'q6.16_HYPER_pcl1', 'q6.17_STRTL_pcl1',
              'intrusion_pcl1', 'avoidance_pcl1', 'hypertention_pcl1', 'depression_pcl1', 'tred_pcl1',
              'q6.1_INTRU_pcl2', 'q6.2_DREAM_pcl2',
              'q6.3_FLASH_pcl2', 'q6.4_UPSET_pcl2',
              'q6.5_PHYS_pcl2', 'q6.6_AVTHT_pcl2', 'q6.7_AVSIT_pcl2', 'q6.8_AMNES_pcl2', 'q6.9_DISINT_pcl2',
              'q6.10_DTACH_pcl2', 'q6.11_NUMB_pcl2', 'q6.12_FUTRE_pcl2', 'q6.13_SLEEP_pcl2',
              'q6.14_ANGER_pcl2', 'q6.15_CONC_pcl2', 'q6.16_HYPER_pcl2', 'q6.17_STRTL_pcl2',
              'intrusion_pcl2', 'avoidance_pcl2', 'hypertention_pcl2', 'depression_pcl2', 'tred_pcl2']
target_features = ["PCL_Strict3"]

ID = ["ID"]


path = "C:\‏‏PycharmProjects\PTSD\Data\PTSD.xlsx"
df = pd.read_excel(path)
df = df[~df['PCL_Strict3'].isna()]
print(df["phq1"].median())
df = df[df["phq1"] >= df["phq1"].median()]
#    df = df[~ ((df["military_exp18_t3"] == 0) & (df["military_exp18_t2"] == 0))]
df = df[features + ID + target_features]
extra_features = 1
if extra_features:
    df_pcl3 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL3.xlsx")
    df_pcl3 = PCL_calculator(df_pcl3)
    df_pcl2 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL2.xlsx")
    df_pcl2 = PCL_calculator(df_pcl2)
    df_pcl1 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL1.xlsx")
    df_pcl1 = PCL_calculator(df_pcl1)

    df = df.merge(df_pcl1, on="ID", how='outer')
    df = df.merge(df_pcl2, suffixes=('_pcl1', '_pcl2'), on="ID", how='outer')
    df = df.merge(df_pcl3.drop(['PCL3_Strict', 'pcl3', 'PCL3_Broad'], axis=1), on="ID", how='outer')

    df = df[~df['PCL_Strict3'].isna()]
    # df = df[~df['tred_cutoff'].isna()]
    df.drop(ID, inplace=True, axis=1)

    all_x_col = features + features_2# + target_features_2
else:
    all_x_col = features

# y_col = ["tred_cutoff"]
y_col = ["PCL_Strict3"]
X = df[all_x_col]
Y = df[y_col].values
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, Y, test_size=0.25, random_state=271828, stratify=Y)
c = ((len(y_train_0) - sum(y_train_0))/ sum(y_train_0))[0]
print(f"c = {c}")
X_train, X_test, y_train, y_test = train_test_split(X_train_0, y_train_0, test_size=0.2, stratify=y_train_0)


pipe_xgb = Pipeline(steps=[
    ('classifier', XGBClassifier())
])

params_xgb = {
    'classifier': [XGBClassifier()],
    'classifier__n_estimators': [100, 250, 400, 700],
    'classifier__max_depth': [2, 3, 6, 10],
    'classifier__learning_rate': [0.1, 0.05, 0.25],
    'classifier__gamma': [0, 0.5, 1],
    # 'classifier__min_child_weight': [1, 0.5, 2],
    'classifier__scale_pos_weight': [c, 2 * c, 0.5 * c],
    # 'classifier__reg_alpha': [0, 1, 0.5],
    # 'classifier__reg_lambda': [0, 1, 0.5]
}
loo = StratifiedKFold(5)

gs_xgb = GridSearchCV(estimator=pipe_xgb,
                      param_grid=params_xgb,
                      scoring='f1',
                      cv=loo)
# Fit grid search
gs_xgb.fit(X_train, y_train.ravel())
# Best params
print('Best params: %s' % gs_xgb.best_params_)
# Best training data accuracy
print('Best training score: %.3f' % gs_xgb.best_score_)
# Predict on test data with best params
y_pred = gs_xgb.predict(X_test)
# Test data accuracy of model with best params
print('Test set score score for best params: %.3f ' % f1_score(y_test, y_pred))
