import numpy as np
from sklearn import preprocessing
import pandas as pd
from DataImporting.visualize_features import auto_visualize_features
from dim_reduction import random_forest_selection
from dim_reduction.PCA_Obj import PCA_Obj
from sklearn.feature_selection import VarianceThreshold
from DataImporting.ImportData import get_data,refactor_labels
from sklearn.pipeline import Pipeline

#paths

path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix.xlsx"

#pca_explained_variance_graph_path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\each_matrix\\pca_explained_variance_graph.jpg"
#features_after_pca = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix_v1.csv"

feature_importance_txt_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\each_matrix"
processed_dataframe_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix_features_processed_v2.csv"

group_column = 'group'
subject_number_column = 'Subject_Number'
random_forest = 0

def feature_selection_pipeline_from_file():
    #get data
    dataset = refactor_labels(get_data(path, 'Sheet1'), group_column)

    # all the visualizations
    #auto_visualize_features(dataset.drop([subject_number_column], 1))

    #remove missing values columns
    non_missing_values_treshold = len(dataset.index) * 0.99
    dataset.dropna(axis=1, thresh=non_missing_values_treshold, inplace=True)

    #impute missing values
    dataset.fillna(dataset.mean(), inplace=True)

    #set X
    X = dataset.drop([group_column, subject_number_column], 1)
    sbj = dataset[subject_number_column]
    Y = dataset[group_column]

    # standartize X
    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X))

    #cutoff by variance
    variance_threshold = 0.03
    variance_cutoff = VarianceThreshold(threshold=variance_threshold)
    variance_cutoff.fit_transform(X)

    print("p1", X.shape)

    #cutoff high correlation
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    X.drop(X.columns[to_drop], 1, inplace=True)

    print("p2",X.shape)

    #save new df
    processed_dataframe = pd.concat([X, Y, sbj], axis=1)
    processed_dataframe.to_csv(processed_dataframe_path)

    #random forest
    if random_forest:
        k_best_features = 31
        feature_importance = random_forest_selection.get_feature_importance(X,Y)
        random_forest_selection.save_feature_importance(feature_importance_txt_path, feature_importance)
        processed_dataframe, X = random_forest_selection.get_k_most_important_features(X,Y,k_best_features,feature_importance)
        processed_dataframe.to_csv(processed_dataframe_path)
    print("p4", processed_dataframe.shape)





def create_pipline(features_df, labels):
    pipeline = None
    return pipeline
feature_selection_pipeline_from_file()