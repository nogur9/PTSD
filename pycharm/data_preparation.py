import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os


### defenitaion

def extract_2020_data():
    target_features_2020 = ['T2_Combat_Exp',
                            'T2_PCLtotal',
                            'T2_PCL_33',
                            'T2_PCL_25',
                            'T2_PCL_B',
                            'T2_PCL_TRED']


    features_2020 = ['Bagrut.1',
                     'Attention_Def',
                     'T1_DP_ACC_Threat',
                     'T1_DP_ACC_Neutral',
                     # 'T1_DP_Bias',
                     'T1_PHQ',
                     'T1_PCLtotal',
                     'T1_PCL_B',
                     'T1_DP_RT_Threat',
                     'T1_DP_RT_Neutral',
                     'T1_DP_ABV'
                     ]
    df = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2018\נתונים מעובדים - כל המדידות.xlsx")

    df_PCL1 = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2018\מעודכן 19.8.19 קידוד פעימה 1.xlsx", sheet_name="PCL")
    df_PCL1 = df_PCL1.drop(['Notes','Average','Final_Sum', 'PCL_SUM', 'Coding_Dates', 'Coder'], axis=1)
    rename_pcl1 = {i: f'{i}_T1' for i in df_PCL1.columns}
    rename_pcl1.pop('ID')
    df_PCL1 = df_PCL1.rename(rename_pcl1, axis=1)
    df_PCL1 = df_PCL1.dropna(how="all")

    df_PHQ1 = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2018\מעודכן 19.8.19 קידוד פעימה 1.xlsx", sheet_name="PHQ")
    df_PHQ1 = df_PHQ1.drop(['Notes','Average','Final_SUM', 'Coder', 'Coding_Date', 'PHQ_SUM'], axis=1)
    df_PHQ1 = df_PHQ1.dropna(how="all")

    df_PCL3 = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2018\קידוד פעימה 2.xlsx", sheet_name="PCL")
    df_PCL3 = df_PCL3.drop(['Coding_Dates', 'Coder', 'PCL_SUM', 'Notes', 'Average', 'Final_Sum', 'B_AVG',
           'B', 'C_AVG', 'C', 'D_AVG', 'D', 'E_AVG', 'E', 'TRED_AVG', 'TRED', 'Unnamed: 37'], axis=1)
    rename_pcl3 = {i: f'{i}_T3' for i in df_PCL3.columns}
    rename_pcl3.pop('ID')
    df_PCL3 = df_PCL3.rename(rename_pcl3, axis=1)
    df_PCL3 = df_PCL3.dropna(how="all")

    df = df.merge(df_PCL1, on="ID", how = "outer")
    df = df.merge(df_PCL3, on="ID", how = "outer")
    df_2020 = df.merge(df_PHQ1, on="ID", how = "outer")

    return df_2020, target_features_2020, features_2020

# ### 2016


def extract_2016_data():

    target_features_2016 = ['Intrusion_T4', 'Avoidance_T4', 'Hyper_T4', 'PCL_T4']
    features_2016 = [
        'bagrut', 'ADHD', 'RT_threat_NT_T1',
        'RT_neutral_NT_T1',
        'Accuracy_threat_T1', 'Accuracy_NT_T1',
        'Threat_Bias_T1', 'ABV_T1', 'PHQ_T1',
        'Trait_T1', 'State_T1',
        'PCL_T1', 'Intrusion_T1',
        'Avoidance_T1', 'Hyper_T1']

    invalid_ids = [2192]
    data_path = r'C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2016'
    df = pd.read_csv(os.path.join(data_path, "IDF_ABM_16.2.15_wide.csv"))

    df_PCL = pd.read_csv(os.path.join(data_path, r"quesionnaires\questionnaire5_PCL.csv"))
    df_phq9 = pd.read_csv(os.path.join(data_path, r"quesionnaires\questionnaire4_PHQ9.csv"))

    df_PCL = df_PCL.drop(['VAR00002', 'PrimaryLast'],axis=1)
    df_PCL = df_PCL.dropna(subset=df_PCL.drop(['ID'], axis=1).columns, how='all')

    df_phq9 = df_phq9.drop(['Unnamed: 0', 'VAR00001', 'VAR00002',
                            'PrimaryLast', 'VAR00003', 'VAR00004',
                            'VAR00005', 'VAR00006', 'VAR00007', 'PrimaryLast1'], axis=1)

    df_phq9 = df_phq9.dropna(subset = df_phq9.drop(['ID'], axis=1).columns, how='all')

    df = df.merge(df_PCL, on="ID", how="outer")
    df = df.merge(df_phq9, on="ID", how="outer")

    df = df[~ df["ID"].isin(invalid_ids)]
    df = df.drop_duplicates(subset=["ID"])

    df_2016 = df[df.Group == 'control']
    return df_2016, target_features_2016, features_2016

# ### 2009


def extract_2009_data():
    features_2009 = ['highschool_diploma',
                     'ADHD',
                     'phq1',
                     #'trait1',
                     #'state1',
                     'PCL1',
                     'intrusion_PCL_T1',
                     'PCL_Strict1',
                     'ABV',
                     "dyslexia",
                    "ID",
                     "trauma_history6_1",
                     #"terror_p1", "terror_i1", "mva_p1", "mva_i1", "violent1", "sexual1",
                     'T1std1t', 'T1median1t', 'T1std1n', 'T1median1n',
                     'T1mean1t', 'T1mean1n',
                     'T1Acc1t', 'T1Acc1n',
                     'T1bias',
                     # 'T1mean2', 'T1std2', 'T1median2', 'T1mean2.1', 'T1std2.1', 'T1median2.1', 'T1mean2.2', 'T1std2.2',
                     # 'T1median2.2', 'T1mean2.3', 'T1std2.3', 'T1median2.3', 'T1mean2.4', 'T1std2.4', 'T1median2.4',
                     # 'T1mean2.5', 'T1std2.5', 'T1median2.5', 'T1mean2.6', 'T1std2.6', 'T1median2.6', 'T1mean2.7',
                     # 'T1std2.7', 'T1median2.7', 'T1mean2.8', 'T1std2.8', 'T1median2.8',
                     # 'T1acc3t', 'T1mean3t', 'T1std3t', 'T1median3t', 'T1acc3n', 'T1mean3n', 'T1std3n', 'T1median3n'
                     ]
    T1_features = [
                     'phq1',
                     'PCL1']
    cogni = [
             'T1mean1t', 'T1mean1n',
             'T1Acc1t','T1Acc1n',
                     'T1bias', ]
    target_features_2009 = ['PCL3',
                            'PCL_Broad3',
                            'PCL_Strict3']

    df = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\PTSD.xlsx")

    abv = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\ABV\raw data - T1.xlsx", sheet_name='ABV')
    df = df.merge(abv, left_on='ID', right_on='Subject', how='outer')

    rt1 = pd.read_csv(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\RT_data\duvdevan_1_4_09_combined.csv")
    rt2 = pd.read_csv(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\RT_data\golani combined.csv")
    rt3 = pd.read_csv(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\RT_data\tzanchanim I_7_12_08_combined.csv")
    cogs = ['T1std1t', 'T1median1t', 'T1mean1t', 'T1mean1n', 'T1std1n', 'T1median1n','T1mean2', 'T1std2', 'T1median2', 'T1mean2.1', 'T1std2.1', 'T1median2.1', 'T1mean2.2', 'T1std2.2',
                     'T1median2.2', 'T1mean2.3', 'T1std2.3', 'T1median2.3', 'T1mean2.4', 'T1std2.4', 'T1median2.4',
                     'T1mean2.5', 'T1std2.5', 'T1median2.5', 'T1mean2.6', 'T1std2.6', 'T1median2.6', 'T1mean2.7',
                     'T1std2.7', 'T1median2.7', 'T1mean2.8', 'T1std2.8', 'T1median2.8',
                     'T1acc3t', 'T1mean3t', 'T1std3t', 'T1median3t', 'T1acc3n', 'T1mean3n', 'T1std3n', 'T1median3n',
                      'ID']
    rts = pd.concat([rt3[cogs], rt2[cogs], rt1[cogs]])
    rts = rts.drop_duplicates("ID")
    df = df.merge(rts[cogs], on="ID", how="outer")

    pcl_T1 = pd.read_csv(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\questionnaires\questionnaire_PCL1.csv")
    pcl_T3 = pd.read_excel(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\questionnaires\questionnaire6PCL3.xlsx")
    phq9 = pd.read_csv(r"C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009\questionnaires\questionnaire5_PHQ9.csv")

    rename_PCL_T3 = {i: f'{i}_PCL_T3' for i in pcl_T3.columns}
    rename_PCL_T3.pop('ID')
    pcl_T3 = pcl_T3.drop(['pcl3', 'PCL3_Broad', 'PCL3_Strict'], axis=1)
    pcl_T3 = pcl_T3.rename(rename_PCL_T3, axis=1)

    rename_PCL_T1 = {i: f'{i}_PCL_T1' for i in pcl_T1.columns}
    rename_PCL_T1.pop('ID')
    pcl_T1 = pcl_T1.drop(['pcl1', 'PCL1_Broad', 'PCL1_Strict'], axis=1)
    pcl_T1 = pcl_T1.rename(rename_PCL_T1, axis=1)

    phq9 = phq9.drop(['Unnamed: 0', 'T1depression', 'T1depression_new', 'T1PHQ_9'], axis=1)
    rename_PHQ_T1 = {i: f'{i[2::]}_PHQ_T1' for i in phq9.columns}
    rename_PHQ_T1.pop('ID')
    phq9 = phq9.rename(rename_PHQ_T1, axis=1)

    df = df.merge(pcl_T3, on="ID", how="outer")
    df = df.merge(pcl_T1, on="ID", how="outer")
    df = df.merge(phq9, on="ID", how="outer")

    df = df.dropna(subset=T1_features, how='all')
    df_2009 = df.dropna(subset=cogni, how='all')

    return df_2009, target_features_2009, features_2009



def extract_data():
    df_2020, target_features_2020, features_2020 = extract_2020_data()
    df_2016, target_features_2016, features_2016 = extract_2016_data()
    df_2009, target_features_2009, features_2009 = extract_2009_data()
    dfs = [df_2020, df_2016, df_2009]
    target_features = [target_features_2020, target_features_2016, target_features_2009]
    features = [features_2020, features_2016, features_2009]
    return dfs, target_features, features
