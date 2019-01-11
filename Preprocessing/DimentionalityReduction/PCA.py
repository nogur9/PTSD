import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataImporting.ImportData import get_data,refactor_labels
from DataImporting.Data_Imputation import imputing_avarage
from sklearn.decomposition import RandomizedPCA


def run_stuff ():
    dataset = refactor_labels(get_data("C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_corr.xlsx", "Sheet1"),"group")
    dataset = imputing_avarage(dataset)
    features_df = dataset.drop(['Age','group','PHQ9','Subject_Number'],1)
    X = features_df.values
    X = StandardScaler().fit_transform(X)

    #X = array[:,3:116]
    pca = RandomizedPCA(50)
    pca.fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()
    # plt.close()
    #
    # y = dataset["group"]
    # k = 10
    # pca = PCA(n_components=k)
    # pca.fit(X)
    # G = pca.transform(X)
    # for l in range(1,k):
    #     for j in range(0,l):
    #         dim1 = [G[i][l] for i in range(len(X))]
    #         dim2 = [G[i][j] for i in range(len(X))]
    #         dim1_zero_vals = [dim1[i] for i in range(len(y)) if y[i] == 0.0]
    #         dim2_zero_vals = [dim2[i] for i in range(len(y)) if y[i] == 0.0]
    #         dim1_one_values = [dim1[i] for i in range(len(y)) if y[i] == 1.0]
    #         dim2_one_values = [dim2[i] for i in range(len(y)) if y[i] == 1.0]
    #         plt.scatter(dim1_zero_vals, dim2_zero_vals, c='r')
    #         plt.scatter(dim1_one_values,dim2_one_values, c='b')
    #         #plt.scatter(dim1,dim2)
    #         plt.show()


def PCA_transforme (df, k, header = 1):
    features_df = imputing_avarage(df)
    if header:
        features_df = df.drop(['Age', 'group', 'PHQ9', 'Subject_Number'], 1)
    X = features_df.values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=k)
    pca.fit(X)
    return pca.transform(X)

def meow ():
    dataset = refactor_labels(get_data("C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_corr.xlsx", "Sheet1"),"group")
    return PCA_transforme(dataset,6)
#run_stuff()



