import xlsxwriter
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE, SelectFdr
import string
import datawig
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from itertools import combinations
import pandas as pd
from fancyimpute import KNN, IterativeImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class EDASingleFeatureBackend:

    # backend of single feature analysis
    dataset_path = r"../Data/PTSD.xlsx"

    regression_target_column = "PCL3"

    def __init__(self, df, feature, targets, impute_missing_values=1, plot_path=r"Visualization/plots", knn=0):
        self.targets = targets
        self.df = df
        self.feature = feature
        self.knn = knn
        # add the missing number to the data
        self.data_for_output_file = {"Feature name": feature, "Existing values count": (~self.df[feature].isnull()).sum(),
                                     "Missing values count": self.df[feature].isnull().sum()}

        # add the missing number to the data
        if impute_missing_values:
            if self.knn:
                self.plot_path = os.path.join(plot_path, 'knn_imputation')
                self.df = self.impute(self.df)
            # if impute_missing_values is an lambda function of imputation method fill the values with it
            else:
                self.plot_path = os.path.join(plot_path, 'mice_imputation')
                self.df = self.impute(self.df)
        else:
            # else, remove the missing values and the corresponding targets
            self.df = self.df[~self.df[feature].isna()]
            self.plot_path = os.path.join(plot_path, 'dropna')
        print("df.shape", df.shape)
        self.X = self.df[feature]

    def plot_distribution(self):

        bins = numpy.linspace(min(self.X), max(self.X), 100)

        # histogram distributation
        plt.hist(self.X, bins, alpha=0.7, color='black')
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

            plt.savefig(os.path.join(self.plot_path, target_column, "target {} feature {}_histogram.png".format(target_column, self.feature)))
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
        self.data_for_output_file["Outliers values"] = outliers.values
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
        if self.knn:
            knn = KNN()
            return pd.DataFrame(knn.fit_transform(df), columns=df.columns)
        else:
            mice = IterativeImputer()
            return pd.DataFrame(mice.fit_transform(df), columns=df.columns)

    def calculate_parameters_of_weak_clf(self):
        # for every target

        X = self.df[[self.feature]]
        for target_column in self.targets:
            Y = self.df[target_column] > 0


            pipe = Pipeline(steps=[
                ('SMOTE', SMOTE(random_state=27)),
                ('classifier', RandomForestClassifier(n_estimators=100))])
            scores = cross_val_score(pipe, X, Y, scoring='f1', cv=5)
            self.data_for_output_file[f"Train f1 score {target_column}"] = sum(scores) / len(scores)



    def get_data_sample(self):
        self.data_for_output_file["Sample"] = self.df.sample(3)

    def write_results(self):
        return self.data_for_output_file


class EDAMultiFeatureBackend:

    def __init__(self, df, features, target, plot_path=r"Visualization/plots", impute_missing_values=1, knn=1):

        self.file_path = os.path.join("feature_summery", target)
        self.target = target
        df = df[~df[target].isna()]
        self.df = df
        self.knn = knn
        self.features = features
        # add the missing number to the data
        if impute_missing_values:
            if self.knn:
            # if impute_missing_values is an lambda function of imputation method fill the values with it
                self.plot_path = os.path.join(plot_path, 'knn_imputation', target)
                self.df = self.impute(self.df)
                self.inteactions_file_name = "sum interaction EDA multiple Features outer merge KNN imputation.xlsx"
            else:
                self.plot_path = os.path.join(plot_path, 'mice_imputation', target)
                self.df = self.impute(self.df)
                self.inteactions_file_name = "EDA multiple Features outer merge mice imputation.xlsx"
        else:
            # else, remove the missing values and the corresponding targets
            self.df = df[features + [target]]
            self.df = self.df.dropna()
            self.plot_path = os.path.join(plot_path, 'dropna', target)
            self.inteactions_file_name = "EDA multiple Features inner merge no imputation.xlsx"
        print("df.shape", df.shape)



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
        with open(os.path.join(self.plot_path, "high_correlations.txt"), "w") as f:
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


        for i in self.features:
            for j in self.features:
                red_patch = mpatches.Patch(color='red', label='negative')
                blue_patch = mpatches.Patch(color='blue', label='positive')
                plt.legend(handles=[red_patch, blue_patch])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.scatter(negative_group[i], negative_group[j], c='r')
                plt.scatter(positive_group[i], positive_group[j], c='b')

                plt.savefig(os.path.join(self.plot_path, "two_features_plots_{}_{}.png".format(i, j)))
                plt.close()


                red_patch = mpatches.Patch(color='red', label='negative')
                blue_patch = mpatches.Patch(color='blue', label='positive')
                plt.legend(handles=[red_patch, blue_patch])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.scatter(positive_group[i], positive_group[j], c='b')
                plt.scatter(negative_group[i], negative_group[j], c='r')
                plt.savefig(os.path.join(self.plot_path, "reversed_two_features_plots_{}_{}.png".format(i, j)))
                plt.close()

    def interactions(self):
        interation_for_output_file = []

        index = 0
        for j in range(2, 3):#len(self.features)//2):
            for i in list(combinations(self.features, j)):
                print(index)

                single_interation_data = {"Features names": i,
                    "Pearson correlation":
                    np.corrcoef(self.df[i[0]] * self.df[i[1]], self.df[self.target])[1, 0]}

                X = (self.df[i[0]] + self.df[i[1]]).values.reshape(-1, 1)
                Y = self.df[self.target]

                pipe = Pipeline(steps=[
                        #('SMOTE', SMOTE(random_state=27)),
                        ('classifier', XGBClassifier(scale_pos_weight=12))])
                scores = cross_val_score(pipe, X, Y, scoring='f1', cv=StratifiedKFold(5))
                single_interation_data[f"Train f1 score"] = sum(scores)/len(scores)
                interation_for_output_file.append(single_interation_data)
                index += 1

        workbook = xlsxwriter.Workbook(os.path.join(self.file_path, self.inteactions_file_name))
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
        if self.knn:
            knn = KNN()
            return pd.DataFrame(knn.fit_transform(df), columns=df.columns)
        else:
            mice = IterativeImputer()
            return pd.DataFrame(mice.fit_transform(df), columns=df.columns)

    def model_checking(self):

        X = self.df[self.features]
        Y = self.df[self.target]

        pipelines = [
             Pipeline(steps=[
                 ('classifier', BalancedRandomForestClassifier(n_estimators=200))]),

             Pipeline(steps=[
                 # ('rfe', RFE(XGBClassifier(), )),
                 ('classifier', BalancedBaggingClassifier(n_estimators=200))]),

            Pipeline(steps=[
                ('rfe', SMOTE()),
                ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1))]),
            Pipeline(steps=[
                 ('rfe', BorderlineSMOTE()),
                 ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1))]),

            Pipeline(steps=[
                # ('rfe', RFE(XGBClassifier(), )),
                ('classifier', XGBClassifier(n_estimators=1000, scale_pos_weight=3, reg_alpha=1))]),

            Pipeline(steps=[
                ('rfe', RFE(XGBClassifier())),
                ('classifier', XGBClassifier(n_estimators=1000, scale_pos_weight=3, reg_alpha=1))])

        ]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)


        for pipe in pipelines:
            scores = cross_val_score(pipe, X_train.values, y_train, scoring='precision', cv=StratifiedKFold(5))
            print("cross val scores")
            print(sum(scores)/5)
            pipe.fit(X_train.values, y_train.values)
            y_pred = pipe.predict(X_test.values)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            print("test scores")
            print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")



    def illigal_genralization_checking(self, X_test, y_test):

        X = self.df[self.features]
        X_test = X_test[self.features]
        Y = self.df[self.target]
        pipe = Pipeline(steps=[('classifier', XGBClassifier(n_estimators=1000, scale_pos_weight=3, reg_alpha=1))])
        y_test = y_test["intrusion_cutoff"].apply(lambda x: int(x))
        scores = cross_val_score(pipe, X, Y, scoring='precision', cv=StratifiedKFold(5))
        print(self.features)
        print("cross vl scores")
        print(sum(scores)/5)
        pipe.fit(X, Y.values)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")



    def data_wig_impute(self):


        #df_train, df_test = datawig.utils.random_split(train)
        #
        # # Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
             input_columns=['1', '2', '3', '4', '5', '6', '7', 'target'],
        #     # column(s) containing information about the column we want to impute
             output_column='0',  # the column we'd like to impute values for
             output_path='imputer_model'  # stores model data and metrics
         )
        #
        # # Fit an imputer model on the train data
       # imputer.fit(train_df=df_train, num_epochs=50)

        # Impute missing values and return original dataframe with predictions
        #imputed = imputer.predict(df_test)



    def three_models_combined(self, intrusion_features, avoidance_features, hypertension_features):

        self.df = self.df[~self.df['intrusion_cutoff'].isna()]
        self.df = self.df[~self.df['avoidance_cutoff'].isna()]
        self.df = self.df[~self.df['hypertention_cutoff'].isna()]
        print("self.df.shape", self.df.shape)
        X = self.df
        Y = self.df[self.target]# strict
        all_Y = [self.target, "intrusion_cutoff", "avoidance_cutoff", "hypertention_cutoff"]




        X_train, X_test, y_train, y_test = train_test_split(X, self.df[all_Y], test_size=0.25, random_state = 8526566, stratify=Y)

        # intrusion
        X_intrusion = X_train[intrusion_features].values
        y_intrusion = y_train["intrusion_cutoff"].apply(lambda x: int(x))
        pipe_intrusion = Pipeline(steps=[
            ('rfe', BorderlineSMOTE()),
            ('classifier', XGBClassifier(n_estimators=100, reg_alpha=1))])
        scores = cross_val_score(pipe_intrusion, X_intrusion, y_intrusion, scoring='precision', cv=StratifiedKFold(5))
        print(f"intrusion {sum(scores)/5}")
        pipe_intrusion.fit(X_intrusion, y_intrusion)

        # avoidance
        X_avoidance = X_train[avoidance_features].values
        y_avoidance = y_train["avoidance_cutoff"].apply(lambda x: int(x))
        pipe_avoidance = Pipeline(steps=[
            ('classifier', XGBClassifier(n_estimators=100, scale_pos_weight=3, reg_alpha=1))])
        scores = cross_val_score(pipe_avoidance, X_avoidance, y_avoidance, scoring='precision', cv=StratifiedKFold(5))
        print(f"avoidance {sum(scores)/5}")
        pipe_avoidance.fit(X_avoidance, y_avoidance)


        # hypertension
        X_hypertension = X_train[hypertension_features].values
        y_hypertention = y_train["hypertention_cutoff"].apply(lambda x: int(x))
        pipe_hypertension = Pipeline(steps=[
            ('classifier', BalancedBaggingClassifier(n_estimators=100))])
        scores = cross_val_score(pipe_hypertension, X_hypertension, y_hypertention, scoring='precision', cv=StratifiedKFold(5))
        print(f"hypertension {sum(scores)/5}")
        pipe_hypertension.fit(X_hypertension, y_hypertention)

        ## combine three classifiers
        X_test_hypertension = X_test[hypertension_features].values
        X_test_avoidance = X_test[avoidance_features].values
        X_test_intrusion = X_test[intrusion_features].values

        y_pred_hypertension = pipe_hypertension.predict(X_test_hypertension)
        y_pred_avoidance = pipe_avoidance.predict(X_test_avoidance)
        y_pred_intrusion = pipe_intrusion.predict(X_test_intrusion)
        y_pred = (y_pred_hypertension * y_pred_avoidance * y_pred_intrusion)

        y_target = y_test["PCL_Strict3"].apply(lambda x: int(x))

        acc = accuracy_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")


