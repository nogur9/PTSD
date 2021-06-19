# coding=utf-8
import numpy as np
import pandas as pd

from data_preparation import extract_2009_data
from preprocessing import PCL_calculator, cv_preprocessing
import os
import json
from quessionaires import merge_questionnaire_data, get_target_features

def load_data(print_columns=1, extract_t3=False):
    data_path = r'C:\Users\nogag\Documents\birocracy\PTSDClassifier\PTSD\Data\2009'
    df, target_features, features = extract_2009_data()

    # import predefined variables
    variables_path = r"variables.json"

    with open(variables_path) as f:
        variables = json.load(f)
    t1_features = variables['t1_features']
    #features = variables['features']
    questionnaires = ["PCL", "PHQ9"]#, "trait_anxiety_revised", "state_anxiety_revised"]
    questionnaires_features = []

    for questionnaire in questionnaires:
        questionnaires_features.extend(variables['questionnaires'][questionnaire])
        df = merge_questionnaire_data(df, questionnaire, os.path.join(data_path, 'questionnaires'), variables['questionnaires'][questionnaire])

    # get targets

    df, target_features = get_target_features(df, data_path)

    # removing missing Y's

    df = df[~df[target_features[0]].isna()]

    df_preprocessed = df[features + target_features + questionnaires_features]
    ######

    df_t1 = df_preprocessed[t1_features]
    df_preprocessed = df_preprocessed[(df_t1.isna().astype(int).sum(axis=1) < len(t1_features))]
    df_preprocessed["t1_missing"] = df_t1.isna().astype(int).sum(axis=1) > len(t1_features)/2
    df_preprocessed = df_preprocessed[~ df_preprocessed['t1_missing']]
    #features.extend(["t1_missing"])

 #   cv_preprocessing(df_preprocessed, df_preprocessed)
    return df_preprocessed, features + questionnaires_features, target_features



if __name__ == "__main__":
    load_data()
