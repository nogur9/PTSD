import xlsxwriter

import string
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import  combinations
import pandas as pd
from fancyimpute import KNN, IterativeImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class EDASingleFeatureBackend:

    # backend of single feature analysis
    dataset_path = r"../Data/PTSD.xlsx"

    regression_target_column = "PCL3"

    def __init__(self, df, feature, targets, impute_missing_values=1, plot_path=r"Visualization/plots"):
        self.targets = targets
        self.df = df
        self.feature = feature
        self.plot_path = plot_path

        # add the missing number to the data
        self.data_for_output_file = {"Feature name": feature, "Existing values count": (~self.df[feature].isnull()).sum(),
                                     "Missing values count": self.df[feature].isnull().sum()}

        # add the missing number to the data
        if impute_missing_values:
            # if impute_missing_values is an lambda function of imputation method fill the values with it
            self.df = self.impute(self.df)
        else:
            # else, remove the missing values and the corresponding targets
            self.df = self.df[~self.df[feature].isna()]

        self.X = self.df[feature]

    def plot_distribution(self):

        bins = numpy.linspace(min(self.X), max(self.X), 100)

        # histogram distributation
        plt.hist(self.X, bins, alpha=0.7, color='r')
        plt.title(self.feature)

        plt.savefig(os.path.join(self.plot_path, "{}_feature_histogram.png".format(self.feature)))
        plt.close()

        for target_column in self.targets:

            # blue
            positive_group = self.X[self.df[target_column] > 0]

            # red
            negative_group = self.X[self.df[target_column] == 0]

            # negative than positive distributation
            red_patch = mpatches.Patch(color='red', label='negative')
            blue_patch = mpatches.Patch(color='blue', label='positive')
            plt.legend(handles=[red_patch, blue_patch])
            plt.hist(negative_group, bins, alpha=0.7, color='r')
            plt.hist(positive_group, bins, alpha=0.7, color='b')
            plt.title(self.feature)

            plt.savefig(os.path.join(self.plot_path, "{}_{}_histogram_positive_group.png".format(target_column, self.feature)))
            plt.close()

            # positive than negative distributation
            reversed_plot = 0
            if reversed_plot:
                red_patch = mpatches.Patch(color='red', label='negative')
                blue_patch = mpatches.Patch(color='blue', label='positive')
                plt.legend(handles=[red_patch, blue_patch])
                plt.hist(positive_group, bins, alpha=0.7, color='b')
                plt.hist(negative_group, bins, alpha=0.7, color='r')
                plt.title(self.feature)

                plt.savefig(os.path.join(self.plot_path, "{}_feature_histogram_negative_group.png".format(self.feature)))
                plt.close()

            reg_plot = 0
            if reg_plot:
                red_patch = mpatches.Patch(color='red', label='negative')
                blue_patch = mpatches.Patch(color='blue', label='positive')
                plt.legend(handles=[red_patch, blue_patch])
                plt.scatter(negative_group, self.df[self.df[target_column] == 0][self.regression_target_column], alpha=0.7, color='r')
                plt.scatter(positive_group, self.df[self.df[target_column] > 0][self.regression_target_column], alpha=0.7, color='b')
                plt.savefig(os.path.join(self.plot_path, "regression_plot_{}.png".format(self.feature)))
                plt.close()

    def analyse_outliers(self, outlier_threshold=3.5):
        # Outliers + outliers ratio
        outliers = self.X[(np.abs(self.X - self.X.mean()) > (outlier_threshold * self.X.std()))]
        self.data_for_output_file["Outliers values"] = outliers
        self.data_for_output_file["Outliers count"] = outliers.shape[0]

    def get_unique_values(self):
        self.data_for_output_file["number of unique values"] = self.X.unique().shape[0]

    def write_statistic_info(self):
        self.data_for_output_file["Statistics\n min, max, average, median, variance, std"] = self.get_statistic_info(self.X)

        for target_column in self.targets:
            positive_group = self.X[self.df[target_column] > 0]
            self.data_for_output_file[f"target {target_column} Positive group statistics\n min, max, average, median, variance, std"] = self.get_statistic_info(positive_group)

            negative_group = self.X[self.df[target_column] == 0]
            self.data_for_output_file[f"target {target_column} Negative group statistics\n min, max, average, median, variance, std"] = self.get_statistic_info(negative_group)


    @staticmethod
    def get_statistic_info(data):
        # Min, max, distribution, variance, average
        return [data.min(), data.max(), data.mean(), data.median(), data.var(), data.std()]

    def calculate_explained_variance_of_target(self):
        # pearson
        for target_column in self.targets:
            self.data_for_output_file[f"Target {target_column} Pearson correlation"] = np.corrcoef(self.X, self.df[target_column])[1, 0]

    def impute(self, df):
        #knn = KNN()
        #return pd.DataFrame(knn.fit_transform(df), columns=df.columns)
        mice = IterativeImputer()
        return pd.DataFrame(mice.fit_transform(df), columns=df.columns)

    def calculate_parameters_of_weak_clf(self):
        # for every target

        X = self.df[[self.feature]]
        for target_column in self.targets:
            Y = self.df[target_column] > 0

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=271828, stratify=Y)
            y_train = y_train > 0

            pipe = Pipeline(steps=[
                ('SMOTE', SMOTE(random_state=27)),
                ('classifier', DecisionTreeClassifier())])

            params_grid = [
                {
                    'classifier': [DecisionTreeClassifier(), KNeighborsClassifier()],
                    'SMOTE__k_neighbors': [5, 3, 10],

                }]

            gs = GridSearchCV(pipe, params_grid, cv=5, scoring='f1')

            gs.fit(X_train, y_train.values)

            self.data_for_output_file[f"Train f1 score {target_column}"] = gs.best_score_
            self.data_for_output_file[f"grid search params {target_column}"] = gs.best_params_



    def get_data_sample(self):
        self.data_for_output_file["Sample"] = self.df.sample(3)

    def write_results(self):
        return self.data_for_output_file


class EDAMultiFeatureBackend:


    def __init__(self,df , features, target, plot_path=r"Visualization/plots"):

        self.file_path = ""
        self.plot_path = plot_path
        self.target = target
        self.df = df
        self.features = features
        self.df = self.impute(self.df)


    def plot_corr_matrix(self, size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''

        corr = self.df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig(os.path.join(self.plot_path, "correlation_matrix.png"))
        plt.close()

    def create_corr_data_file(self):
        """

        :param dataset: Pandas DataFrame dataset

        creates correlation matrix and prints the correlation that are higher then 0.5
        in comment - printing correlation that are lower then -0.5, but in this data its empty.
        """

        names = list(self.df)
        correlations = self.df.corr()

        high_correlations = \
            [[names[i], names[j], correlations.values[i][j]] for i in range(len(names)) for j in range(len(names))
                if (i != j) and (np.abs(correlations) > 0.5).values[i][j]]

        high_correlations.sort(key=lambda x: x[2])
        with open(os.path.join(self.file_path, "high_correlation.txt"), "w") as f:
            for i in high_correlations:
                f.write(str(i)+"\n")


    def two_features_plots(self):
        """

        :param self.dataset: Pandas DataFrame self.dataset

        Makes scatter plot of each two features.
        """

        # blue
        positive_group = self.df[self.df[self.target] == 1]

        # red
        negative_group = self.df[self.df[self.target] == 0]

        names = list(self.df)
        for i in names:
            for j in names:
                red_patch = mpatches.Patch(color='red', label='negative')
                blue_patch = mpatches.Patch(color='blue', label='positive')
                plt.legend(handles=[red_patch, blue_patch])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.scatter(positive_group[i], positive_group[j], c='b')
                plt.scatter(negative_group[i], negative_group[j], c='r')
                plt.savefig(os.path.join(self.plot_path, "switched_two_features_plots_{}_{}.png".format(i, j)))
                plt.close()


    def interactions(self):
        interation_for_output_file = []

        index = 0
        for i in list(combinations(self.features, 2)):
            print(index)

            single_interation_data = {"Features names": i, f"Pearson correlation":
                np.corrcoef(self.df[i[0]] * self.df[i[1]], self.df[self.target])[1, 0]}

            X = self.df[[*i]]
            Y = self.df[self.target]

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=271828, stratify=Y)


            pipe = Pipeline(steps=[
                    ('SMOTE', SMOTE(random_state=27)),
                    ('classifier', DecisionTreeClassifier())])

            params_grid = [
                    {
                        'classifier': [DecisionTreeClassifier(), KNeighborsClassifier()],
                        'SMOTE__k_neighbors': [5, 3, 10]
                    }]

            gs = GridSearchCV(pipe, params_grid, cv=5, scoring='f1')

            gs.fit(X_train, y_train.values)

            single_interation_data[f"Train f1 score"] = gs.best_score_
            single_interation_data[f"grid search params"] = gs.best_params_
            interation_for_output_file.append(single_interation_data)
            index += 1


        file_path = r"feature_summery"
        file_name = "EDA multiple Features no COMT_Ranked_outer merge.xlsx"
        workbook = xlsxwriter.Workbook(os.path.join(file_path, file_name))
        worksheet = workbook.add_worksheet()
        row = 0
        for output_data in interation_for_output_file:
            col = 0
            for key in output_data.keys():
                if row == 0:
                    worksheet.write(row, col, key)
                    row = 1
                    worksheet.write(row, col, str(output_data[key]))
                    col += 1
                    row = 0
                else:
                    worksheet.write(row, col, str(output_data[key]))
                    col += 1
            row += 1
        workbook.close()

    def impute(self, df):
        #knn = KNN()
        #return pd.DataFrame(knn.fit_transform(df), columns=df.columns)
        mice = IterativeImputer()
        return pd.DataFrame(mice.fit_transform(df), columns=df.columns)



