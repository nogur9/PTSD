from sklearn.preprocessing import Imputer
from DataImporting import knn_imputer


def imputing_avarage(dataset):
    '''

    :param dataset: pandas DataFrame dataset. 
    :return: The same dataset where the missing values are replaced with the column's mean value
    '''

    dataset.fillna(dataset.mean(), inplace=True)
    return dataset


def imputing_median(dataset):
    '''

    :param dataset: pandas DataFrame dataset. 
    :return: The same dataset where the missing values are replaced with the column's median value
    '''
    dataset.fillna(dataset.median(), inplace=True)
    return dataset


def imputing_most_frequent(dataset):
    '''

    :param dataset: pandas DataFrame dataset. 
    :return: The same dataset where the missing values are replaced with the column's most common value
    '''

    imp = Imputer(missing_values='NaN', strategy='most_frequent', copy=False)
    imp.fit_transform(dataset)
    return dataset


def imputing_knn(dataset):
    '''

    :param dataset: pandas DataFrame dataset. 
    :return: The same dataset where the missing values are replaced with the predictions of KNN Classifier
    '''

    impute = knn_imputer.Imputer()
    for column in dataset:
        impute.knn(X=dataset, column=column)

    return dataset

