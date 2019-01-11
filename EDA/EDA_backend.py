import string
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import tree
from sklearn.decomposition import PCA


class EDASingleFeatureBackend:

    # backend of single feature analysis
    dataset_path = r"../Data/PTSD.xlsx"
    target_column = "PCL_Strict3"
    regression_target_column = "PCL3"

    def __init__(self, feature, impute_missing_values=None, plot_path=r"Visualization/plots"):

        self.feature = feature
        self.plot_path = plot_path
        # open the data file with feature and target

        self.df = pd.read_excel(self.dataset_path)

        # add the missing number to the data

        self.data_for_output_file = {"Feature name": feature,
                                     "Missing values count without removing missing target":
                                         self.df[feature].isnull().sum()}

        # remove missing target
        self.df = self.df[~self.df[self.target_column].isna()]

        # add the missing number to the data
        self.data_for_output_file["Missing values count after removing missing target"] =\
            self.df[feature].isnull().sum()

        if impute_missing_values is not None:
            # if impute_missing_values is an lambda function of imputation method fill the values with it
            self.df[feature].fillna(impute_missing_values(self.df[feature]), inplace=True)
        else:
            # else, remove the missing values and the corresponding targets
            self.df = self.df[~self.df[feature].isna()]

        self.X = self.df[feature]

    def plot_distribution(self):
        # blue
        positive_group = self.X[self.df[self.target_column] == 1]

        # red
        negative_group = self.X[self.df[self.target_column] == 0]

        bins = numpy.linspace(min(self.X), max(self.X), 100)

        red_patch = mpatches.Patch(color='red', label='negative')
        blue_patch = mpatches.Patch(color='blue', label='positive')
        plt.legend(handles=[red_patch, blue_patch])
        plt.hist(negative_group, bins, alpha=0.7, color='r')
        plt.hist(positive_group, bins, alpha=0.7, color='b')
        plt.title(self.feature)

        plt.savefig(os.path.join(self.plot_path, "group_hist_{}.png".format(self.feature)))
        plt.close()


        red_patch = mpatches.Patch(color='red', label='negative')
        blue_patch = mpatches.Patch(color='blue', label='positive')
        plt.legend(handles=[red_patch, blue_patch])
        plt.scatter(negative_group, self.df[self.df[self.target_column] == 0][self.regression_target_column], alpha=0.7, color='r')
        plt.scatter(positive_group, self.df[self.df[self.target_column] == 1][self.regression_target_column], alpha=0.7, color='b')
        plt.savefig(os.path.join(self.plot_path, "regression_plot_{}.png".format(self.feature)))
        plt.close()

    def analyse_outliers(self, outlier_threshold=3):

        # Outliers + outliers ratio
        outliers = self.X[(np.abs(self.X - self.X.mean()) > (outlier_threshold * self.X.std()))]
        self.data_for_output_file["Outliers values"] = outliers
        self.data_for_output_file["Outliers count"] = outliers.shape[0]

    def get_unique_values(self):
        self.data_for_output_file["number of unique values"] = self.X.unique().shape[0]

    def write_statistic_info(self):
        self.data_for_output_file["statistics"] = self.get_statistic_info(self.X)

        positive_group = self.X[self.df[self.target_column] == 1]
        self.data_for_output_file["Positive group statistics"] = self.get_statistic_info(positive_group)

        negative_group = self.X[self.df[self.target_column] == 0]
        self.data_for_output_file["Negative group statistics"] = self.get_statistic_info(negative_group)


    @staticmethod
    def get_statistic_info(data):
        # Min, max, distribution, variance, average
        return [data.min(), data.max(), data.mean(), data.var(), data.std()]

    def calculate_explained_variance_of_target(self):
        # pearson
        self.data_for_output_file["Pearson correlation"] = np.corrcoef(self.X, self.df[self.target_column])[1, 0]

    def calculate_parameters_of_weak_clf(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.X.reshape(-1, 1), self.df[self.target_column])
        self.data_for_output_file["Train score tree"] = clf.score(self.X.reshape(-1, 1), self.df[self.target_column])
        self.data_for_output_file["Tree params"] = clf.get_params()

    def get_data_sample(self):
        self.data_for_output_file["Sample"] = self.df.sample(3)

    def write_results(self):
        return self.data_for_output_file


class EDAMultiFeatureBackend:
    # backend of multiple features analysis
    dataset_path = r"../Data/PTSD.xlsx"
    target_column = "PCL_Strict3"
    features_groups = {}
    plot_path = r"Visualization/plots"
    file_path = r"feature_summery"

    binary_features = [ "highschool_diploma", "Hebrew", "dyslexia", "ADHD"]
    categorical_features = ["age",  "ptgi2",
                         "active_coping1", "planning1", "positive_reframing1", "acceptance1", "humor1",
                         "religion1", "emotional_support1", "instrumental_support1", "self_distraction1",
                         "denial1", "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1",
                         "active_coping2", "planning2", "positive_reframing2", "acceptance2", "humor2",
                         "religion2", "emotional_support2", "instumental_support2", "self_distraction2",
                         "denial2", "venting2", "substance_use2", "behavioral_disengagement2", "self_blam2",
                         "trauma_history8_1", "military_exposure_unit", "HML_5HTT", "HL_MAOA", "HML_NPY",
                         "COMT_Hap1_recode", "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5", "Ashken_scale",
                         "Sephar_scale", "Unknown"]

    numerical_features = ["T1Acc1t", "T1Acc1n", "T2Acc1t", "T2Acc1n", "T1ETBE", "T1bias", "T2bias", "state1", "state2",
                           "trait1", "trait2", "lot1", "lot2", "phq1", "phq2", "cd_risc1", "PCL1", "PCL2"]
    missing_values_threshold_for_featues = 0.5
    missing_values_threshold_for_subjects = 0.95

    def __init__(self, df, include_data_without_target=True):

        # open the data file with feature and target

        self.df = df

        self.df = self.df.dropna(axis=1, thresh=self.missing_values_threshold_for_featues)
        self.df = self.df.dropna(axis=0, thresh=self.missing_values_threshold_for_subjects)

        # remove missing target
        if not include_data_without_target:
            self.df = self.df[~self.df[self.target_column].isna()]

        self.df[self.binary_features] = self.df[self.binary_features].fillna(self.df.mode().iloc[0])
        mean_imputation_features = self.categorical_features + self.numerical_features
        self.df[mean_imputation_features] = self.df[mean_imputation_features].fillna(self.df.mean())


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
        positive_group = self.df[self.df[self.target_column] == 1]

        # red
        negative_group = self.df[self.df[self.target_column] == 0]

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
    def plot_features_groups(self):
        pass

    def calculate_clusters(self):
        pass
