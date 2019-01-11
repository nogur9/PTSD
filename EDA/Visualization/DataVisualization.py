import string
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import os
import pandas as pd
from DataImporting.ImportData import refactor_labels
from scipy import stats

class DataVisualizationObj:
    dataset = None
    group_column = ""

    def __init__(self, data_set, group_column="group"):
        self.dataset = refactor_labels(data_set, group_column)
        self.group_column = group_column

    def create_two_hists_by_group(self, path = None):
        df = self.dataset.dropna(axis=1)
        for feature in df:
            x = df[df[self.group_column] == 1][feature]
            #blue
            y = df[df[self.group_column] == 0][feature]
            #red

            bins = numpy.linspace(min(df[feature]),max(df[feature]), 100)
            fig, axs = plt.subplots(2, 1, sharex=True)
            fig.suptitle(feature, fontsize=16)
            axs[0].hist(x, bins=bins, color='b')
            axs[0].set_title("high")
            axs[1].hist(y, bins=bins, color='c')
            axs[1].set_title("low")

            if path:
                plt.savefig(os.path.join(path,"subplot_hist_{}.png".format(self.format_filename(str(feature)))))
            else:
                plt.show()

    def format_filename(self, s):
        """Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.

    Note: this method may produce invalid filenames such as ``, `.` or `..`
    When I use this method I prepend a date string like '2009_01_15_19_46_32_'
    and append a file extension like '.txt', so I avoid the potential of using
    an invalid filename.

    """
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = ''.join(c for c in s if c in valid_chars)
        filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
        return filename

    def create_binary_hist(self, path = None):
        df = self.dataset.dropna(axis=1)
        for feature in df:
            x = df[df[self.group_column] == 1][feature]
            #blue
            y = df[df[self.group_column] == 0][feature]
            #red

            bins = numpy.linspace(min(df[feature]),max(df[feature]), 100)
            red_patch = mpatches.Patch(color='red', label='low')
            blue_patch = mpatches.Patch(color='blue', label='high')
            plt.legend(handles=[red_patch, blue_patch])
            plt.hist(x, bins, alpha=0.7, label='high', color='b')
            plt.hist(y, bins, alpha=0.7, label='low', color='r')
            plt.title(feature)
            if path:
                plt.savefig(os.path.join(path,"binary_hist_{}.png".format(self.format_filename(str(feature)))))
            else:
                plt.show()


    def plot_scatters(self, path=None):
        """
    
        :param self.dataset: Pandas DataFrame self.dataset
    
        Makes scatter plot of each two features.
        """
        df = self.dataset.dropna(axis=1)
        zero_vals = [self.dataset.values[i] for i in range(len(self.dataset.values)) if self.dataset.values[i][0] == 0.0]
        one_values = [self.dataset.values[i] for i in range(len(self.dataset.values)) if self.dataset.values[i][0] == 1.0]
    
        for i in range(1):
            for j in range(29,34):
                red_patch = mpatches.Patch(color='red', label='Non SAD')
                blue_patch = mpatches.Patch(color='blue', label='SAD')
                plt.legend(handles=[red_patch, blue_patch])
                plt.hold(True)
                plt.xlabel(list(self.dataset)[i])
                plt.ylabel(list(self.dataset)[j])
                plt.scatter([float(zero_vals[k][i]) for k in range(len(zero_vals))],
                            [float(zero_vals[k][j]) for k in range(len(zero_vals))], c='r')
                plt.scatter([float(one_values[k][i]) for k in range(len(one_values))],
                            [float(one_values[k][j]) for k in range(len(one_values))], c='b')
                plt.savefig(os.path.join(path, "correlations_figure_{}.png"))


    def box_plot(self):
        """
    
        :param self.dataset: Pandas DataFrame self.dataset
    
        Makes box and whisker plots without the outcome variable.
        """
    
        dataset_without_output = self.dataset.drop(self.group_column, axis=1)
        dataset_without_output.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False, Columns=['Age','PHQ9'])
        plt.show()
        plt.close()


    def plot_data(self):
    
        self.dataset.hist()
        plt.show()
    
        scatter_matrix(self.dataset)
        plt.show()


    def plot_corr(self, size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''

        corr = self.dataset.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()

    def plot_correlation_matrix(self, path=None):
        """

        :param dataset: Pandas DataFrame dataset

        creates correlation matrix and prints the correlation that are higher then 0.5
        in comment - printing correlation that are lower then -0.5, but in this data its empty.
        """
        df = self.dataset.dropna(axis=1)
        names = list(df)
        names.remove('group')
        correlations = df.corr()
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        #correlations = sorted(correlations)
        print("Correlations:")
        all_corr = [[names[i], names[j], correlations.values[i][j]] for i in range(len(names)) for j in range(len(names))]
        high_corr = [[names[i], names[j], correlations.values[i][j]] for i in range(len(names)) for j in range(len(names))
                if (not i == j) and ((correlations > 0.5).values[i][j] or (correlations < -0.5).values[i][j])]
        all_corr.sort(key=lambda x: x[2])
        high_corr.sort(key=lambda x: x[2])
        print(*high_corr, sep="\n")

        fig.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.title('Correlations matrix')
        plt.savefig(os.path.join(path, "correlations_figure.png"))
        if path:
            with open(os.path.join(path, "high_correlation.txt"),"w") as f:
                for i in high_corr:
                    f.write(str(i)+"\n")
            with open(os.path.join(path, "all_correlation.txt"),"w") as f:
                for i in all_corr:
                    f.write(str(i)+"\n")


    def print_data(self):

        #print(dataset.head(20))
        print(self.dataset.describe())
        print(self.dataset.shape)
        print(self.dataset.groupby(self.group_column).size())


    def print_missing_values(self, path=None):
        print("Missing Values")
        df = self.dataset.dropna(axis=1)
        print(df.shape)
        print(self.dataset.isnull().sum())
        result = [(name,self.dataset[name].isnull().sum()) for name in self.dataset.keys()]
        if path:
            with open(os.path.join(path, "missing.txt"),"w") as f:
                for i in result:
                    f.write(str(i)+"\n")


    def print_variance(self, path=None):
        df = self.dataset.dropna(axis=1)
        df = df.drop('group', 1)
        standard_scaler = preprocessing.MinMaxScaler()
        data = standard_scaler.fit_transform(df)
        selector = VarianceThreshold()
        selector.fit_transform(data)
        result = sorted(zip(list(df), selector.variances_), key=lambda x: x[1])
        print("Variance")
        print(*result, sep="\n")
        if path:
            with open(os.path.join(path, "variance.txt"),"w") as f:
                for i in result:
                    f.write(str(i)+"\n")


    def detect_outliers(self, path=None):
        df = self.dataset.dropna(axis=1)
        i, j, k, l = (0,0,0,0)
        names = list(df)
        names.remove('group')
        for name in names:
            tmp_df = df[(np.abs(df[name] - df[name].mean()) > (3 * df[name].std()))]
            #print(tmp_df)
            #print(name,"\n", tmp_df.shape, "\n", tmp_df.Subject_Number)


    def describe(self):
        print(self.dataset.shape)
        print("separator")
        print(self.dataset.describe())
        print("separator")
        print(self.dataset.groupby(self.group_column).describe())