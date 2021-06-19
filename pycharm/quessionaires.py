import scipy
import pandas as pd
from preprocessing import PCL_calculator, PHQ_calculator
import os


def merge_questionnaire_data(df, questionnaire, data_path, vars):
    df_questionnaire = pd.read_csv(os.path.join(data_path, fr"questionnaire_{questionnaire}.csv"))
    if questionnaire == "PCL":
        df_questionnaire = PCL_calculator(df_questionnaire)
    elif questionnaire == 'PHQ9':
        df_questionnaire = PHQ_calculator(df_questionnaire)
    df = df.merge(df_questionnaire[vars + ["ID"]], on="ID", how='outer')
    return df


def get_target_features(df, data_path):
    df_pcl3 = pd.read_excel(os.path.join(data_path, r"questionnaires\questionnaire6PCL3.xlsx"))
    df_pcl3 = PCL_calculator(df_pcl3)
    ts = ["q6.1_INTRU", "q6.2_DREAM", "q6.3_FLASH", "q6.4_UPSET", "q6.5_PHYS"]
    df_pcl3[["target_tred", "target_intrusion", "target_avoidance", "target_hyper", "target_binary_intrusion", "target_binary_tred", "target_binary_avoidance", "target_binary_hyper"] + [f'{i}_target' for i in ts]] = df_pcl3[["tred_score", "intrusion_score", "avoidance_score", "hyper_score", "binary_intrusion", "binary_tred", "binary_avoidance", "binary_hyper"] + ts]
    df = df.merge(df_pcl3[["target_tred", "target_intrusion", "target_avoidance", "target_hyper", "target_binary_intrusion", "target_binary_tred", "target_binary_avoidance", "target_binary_hyper", 'ID'] + [f'{i}_target' for i in ts]], on="ID", how='outer')
    target_features = ['PCL_Strict3', "PCL3", "target_tred", "target_intrusion", "target_avoidance", "target_hyper", "phq3", "target_binary_intrusion", "target_binary_tred", "target_binary_avoidance", "target_binary_hyper"] + [f'{i}_target' for i in ts]
    return df, target_features


