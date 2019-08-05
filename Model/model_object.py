from fancyimpute import IterativeImputer

from Model.model_object_backend import EDAMultiFeatureBackend
import pandas as pd
import xlsxwriter
import os
from feature_engineering.PCL_calculator import PCL_calculator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#.EDASingleFeatureBackend, EDA_backend.EDAMultiFeatureBackend
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

mew = 1
class Model:

    features = ["age", "highschool_diploma", "dyslexia", "ADHD", "T1Acc1t", "T1Acc1n", "T1bias", "phq1", "lot1",
                "trait1",
                "state1", "PCL1", "PCL_Broad1", "PCL_Strict1", "phq2", "lot2", "trait2", "state2", "PCL2", "PCL_Broad2",
                "PCL_Strict2", "cd_risc1", "active_coping1", "planning1", "positive_reframing1", "acceptance1",
                "humor1",
                "religion1", "emotional_support1", "instrumental_support1", "self_distraction1", "denial1",
                "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1", "active_coping2", "planning2",
                "positive_reframing2", "acceptance2", "humor2", "religion2", "emotional_support2",
                "instrumental_support2",
                "self_distraction2", "denial2", "venting2", "substance_use2", "behavioral_disengagement2",
                "self_blame2",
                "trauma_history8_1", "HML_5HTT", "HL_MAOA", "HML_NPY", "COMT_Hap1_recode",
                "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5"]

    features_2 = ['q6.1_INTRU_pcl1', 'q6.2_DREAM_pcl1',
                  'q6.3_FLASH_pcl1', 'q6.4_UPSET_pcl1',
                  'q6.5_PHYS_pcl1', 'q6.6_AVTHT_pcl1', 'q6.7_AVSIT_pcl1', 'q6.8_AMNES_pcl1', 'q6.9_DISINT_pcl1',
                  'q6.10_DTACH_pcl1', 'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl1', 'q6.13_SLEEP_pcl1',
                  'q6.14_ANGER_pcl1', 'q6.15_CONC_pcl1', 'q6.16_HYPER_pcl1', 'q6.17_STRTL_pcl1',
                  'intrusion_pcl1', 'avoidance_pcl1', 'hypertention_pcl1', 'depression_pcl1', 'tred_pcl1',
                  'q6.1_INTRU_pcl2', 'q6.2_DREAM_pcl2',
                  'q6.3_FLASH_pcl2', 'q6.4_UPSET_pcl2',
                  'q6.5_PHYS_pcl2', 'q6.6_AVTHT_pcl2', 'q6.7_AVSIT_pcl2', 'q6.8_AMNES_pcl2', 'q6.9_DISINT_pcl2',
                  'q6.10_DTACH_pcl2', 'q6.11_NUMB_pcl2', 'q6.12_FUTRE_pcl2', 'q6.13_SLEEP_pcl2',
                  'q6.14_ANGER_pcl2', 'q6.15_CONC_pcl2', 'q6.16_HYPER_pcl2', 'q6.17_STRTL_pcl2',
                  'intrusion_pcl2', 'avoidance_pcl2', 'hypertention_pcl2', 'depression_pcl2', 'tred_pcl2']
    target_features = ["PCL_Strict3", "PCL3"]
    target_features_2 = ["intrusion_cutoff", "avoidance_cutoff", "hypertention_cutoff",
                         'depression_cutoff', 'diagnosis', "PCL3", "only_avoidance_cutoff", "tred_cutoff",
                         "regression_cutoff_33", "regression_cutoff_50"]
    ID = ["ID"]
    dataset_path = r"../Data/PTSD.xlsx"

    multiple_features_no_imputation = ['q6.16_HYPER_pcl1',  'hypertention_pcl2', 'q6.5_PHYS_pcl2', 'q6.12_FUTRE_pcl1',
                                       'cd_risc1',  'q6.2_DREAM_pcl2',  'q6.14_ANGER_pcl2', 'positive_reframing2',
                                       'venting2', 'q6.15_CONC_pcl1', 'q6.8_AMNES_pcl1',
                                       'q6.15_CONC_pcl2', 'PCL_Broad2', 'phq2', 'q6.4_UPSET_pcl2']

    def __init__(self):
        path = "C:\‏‏PycharmProjects\PTSD\Data\PTSD.xlsx"
        df = pd.read_excel(path)
        df = df[~df['PCL_Strict3'].isna()]
        df = df[~ ((df["military_exp18_t3"] == 0) & (df["military_exp18_t2"] == 0))]
        df = df[self.features + self.ID + self.target_features]
        df_pcl3 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL3.xlsx")
        df_pcl3 = PCL_calculator(df_pcl3)
        df_pcl2 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL2.xlsx")
        df_pcl2 = PCL_calculator(df_pcl2)
        df_pcl1 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL1.xlsx")
        df_pcl1 = PCL_calculator(df_pcl1)

        df = df.merge(df_pcl1, on="ID", how='outer')
        df = df.merge(df_pcl2, suffixes=('_pcl1', '_pcl2'), on="ID", how='outer')
        df = df.merge(df_pcl3.drop(['PCL3_Strict', 'pcl3', 'PCL3_Broad'], axis=1), on="ID", how='outer')

        df = df[~df['PCL_Strict3'].isna()]
        #df = df[~df['tred_cutoff'].isna()]
        df.drop(self.ID, inplace=True, axis=1)
        if mew:
            mice = IterativeImputer()
            df = pd.DataFrame(mice.fit_transform(df), columns=df.columns)

        all_x_col = self.features + self.features_2 + self.target_features_2
        #all_x_col = self.features + self.features_2
        #y_col = ["tred_cutoff"]
        y_col = ["PCL_Strict3"]
        X = df[all_x_col]
        Y = df[y_col]
        if mew:
            X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, Y, test_size=0.25, random_state=271828, stratify=Y)
            X_train, X_test, y_train, y_test = train_test_split(X_train_0, y_train_0, test_size=0.25, random_state=271828, stratify=y_train_0)
            df = pd.concat([X_train, y_train], axis=1)
            self.X_test = X_test
            self.y_test =y_test

            self.X_train_0 = X_train_0
            self.X_test_0 = X_test_0
            self.y_train_0 = y_train_0
            self.y_test_0 = y_test_0
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                                random_state=271828, stratify=Y)
            df = pd.concat([X_train, y_train], axis=1)
            self.X_test = X_test
            self.y_test = y_test

        self.df = df

    def test_models_for_targets(self):
        targets = {
            'intrusion_cutoff': 1,
            'avoidance_cutoff': 1,
            'hypertention_cutoff': 1,
            'depression_cutoff': 1,
            'only_avoidance_cutoff': 1,
            'diagnosis': 1,
            'regression_cutoff_33': 1,
            'regression_cutoff_50': 1,
            'tred_cutoff': 1
        }
        targets_list = [i for i in targets if targets[i] == 1]
        for target in targets_list:
            print(f"\n\n\n\t\b{target}\n")
            multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
            multiple_features_eda.model_selection_by_grid_search()



    def test_models_with_LOO(self):
        targets = {
            'intrusion_cutoff': 1,
            'avoidance_cutoff': 1,
            'hypertention_cutoff': 1,
            'depression_cutoff': 1,
            'only_avoidance_cutoff': 1,
            'diagnosis': 1,
            'regression_cutoff_33': 1,
            'regression_cutoff_50': 1,
            'tred_cutoff': 1
        }
        targets_list = [i for i in targets if targets[i] == 1]
        for target in targets_list:
            print(f"\n\n\n\t\b{target}\n")
            multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
            multiple_features_eda.model_selection_by_grid_search_loo()

    def test_algorithms_for_trget_intrusion(self):
        print("precision score")
        print("\n\n\n intrusion_cutoff \n")
        target = "intrusion_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n intrusion_cutoff with resample\nn=10")
        multiple_features_eda.model_checking(10, scoring='precision')
        print("\n intrusion_cutoff without resample\nn=10")
        multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n intrusion_cutoff with resample\nn=20")
        multiple_features_eda.model_checking(20, scoring='precision')
        print("\n intrusion_cutoff without resample\nn=20")
        multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n intrusion_cutoff with resample\nn=30")
        multiple_features_eda.model_checking(30, scoring='precision')
        print("\n intrusion_cutoff without resample\nn=30")
        multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("intrusion_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("intrusion_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')


    def test_algorithms_for_trget_avoidance(self):
        print("precision score")
        print("\n\n\n avoidance_cutoff \n")
        target = "avoidance_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n avoidance_cutoff with resample\nn=10")
        multiple_features_eda.model_checking(10, scoring='precision')
        print("\n avoidance_cutoff without resample\nn=10")
        multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n avoidance_cutoff with resample\nn=20")
        multiple_features_eda.model_checking(20, scoring='precision')
        print("\n avoidance_cutoff without resample\nn=20")
        multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n avoidance_cutoff with resample\nn=30")
        multiple_features_eda.model_checking(30, scoring='precision')
        print("\n avoidance_cutoff without resample\nn=30")
        multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("avoidance_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("avoidance_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')



    def test_algorithms_for_trget_hypertension(self):
        print("precision score")
        print("\n\n\n hypertention_cutoff \n")
        target = "hypertention_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n hypertention_cutoff with resample\nn=10")
        multiple_features_eda.model_checking(10, scoring='precision')
        print("\n hypertention_cutoff without resample\nn=10")
        multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n hypertention_cutoff with resample\nn=20")
        multiple_features_eda.model_checking(20, scoring='precision')
        print("\n hypertention_cutoff without resample\nn=20")
        multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n hypertention_cutoff with resample\nn=30")
        multiple_features_eda.model_checking(30, scoring='precision')
        print("\n hypertention_cutoff without resample\nn=30")
        multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("hypertention_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("hypertention_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')


    def test_algorithms_for_target_depression(self):
        print("precision score")
        print("\n\n\n depression_cutoff \n")
        target = "depression_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n depression_cutoff with resample\nn=10")
        multiple_features_eda.model_checking(10, scoring='precision')
        print("\n depression_cutoff without resample\nn=10")
        multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n depression_cutoff with resample\nn=20")
        multiple_features_eda.model_checking(20, scoring='precision')
        print("\n depression_cutoff without resample\nn=20")
        multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n depression_cutoff with resample\nn=30")
        multiple_features_eda.model_checking(30, scoring='precision')
        print("\n depression_cutoff without resample\nn=30")
        multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("depression_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("depression_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')

    def test_algorithms_for_target_diagnosis(self):
        print("precision score")
        print("\n\n\n diagnosis \n")
        target = "diagnosis"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n diagnosis with resample\nn=10")
        multiple_features_eda.model_checking(10, scoring='precision')
        print("\n diagnosis without resample\nn=10")
        multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n diagnosis with resample\nn=20")
        multiple_features_eda.model_checking(20, scoring='precision')
        print("\n diagnosis without resample\nn=20")
        multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n diagnosis with resample\nn=30")
        multiple_features_eda.model_checking(30, scoring='precision')
        print("\n diagnosis without resample\nn=30")
        multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        # print("diagnosis logreg f1 score")
        # multiple_features_eda.logistic_regression_grid_search()
        # print("diagnosis logreg precision score")
        # multiple_features_eda.logistic_regression_grid_search(scoring='precision')


    def test_algorithms_for_target_only_avoidance_cutoff(self):
        print("precision score")
        print("\n\n\n only_avoidance_cutoff \n")
        target = "only_avoidance_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n only_avoidance_cutoff with resample\nn=10")
        #multiple_features_eda.model_checking(10, scoring='precision')
        print("\n only_avoidance_cutoff without resample\nn=10")
        #multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n only_avoidance_cutoff with resample\nn=20")
        #multiple_features_eda.model_checking(20, scoring='precision')
        print("\n only_avoidance_cutoff without resample\nn=20")
        #multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n only_avoidance_cutoff with resample\nn=30")
        #multiple_features_eda.model_checking(30, scoring='precision')
        print("\n only_avoidance_cutoff without resample\nn=30")
        #multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("only_avoidance_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("only_avoidance_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')

    def test_algorithms_for_target_regression_cutoff_33(self):
        print("precision score")
        print("\n\n\n regression_cutoff_33 \n")
        target = "regression_cutoff_33"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n regression_cutoff_33 with resample\nn=10")
        #multiple_features_eda.model_checking(10, scoring='precision')
        print("\n regression_cutoff_33 without resample\nn=10")
        #multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n regression_cutoff_33 with resample\nn=20")
        #multiple_features_eda.model_checking(20, scoring='precision')
        print("\n regression_cutoff_33 without resample\nn=20")
        #multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n regression_cutoff_33 with resample\nn=30")
        #multiple_features_eda.model_checking(30, scoring='precision')
        print("\n regression_cutoff_33 without resample\nn=30")
        #multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("regression_cutoff_33 logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("regression_cutoff_33 logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')

    def test_algorithms_for_target_regression_cutoff_50(self):
        print("precision score")
        print("\n\n\n regression_cutoff_50 \n")
        target = "regression_cutoff_50"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n regression_cutoff_50 with resample\nn=10")
        #multiple_features_eda.model_checking(10, scoring='precision')
        print("\n regression_cutoff_50 without resample\nn=10")
        #multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n regression_cutoff_50 with resample\nn=20")
        #multiple_features_eda.model_checking(20, scoring='precision')
        print("\n regression_cutoff_50 without resample\nn=20")
        #multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n regression_cutoff_50 with resample\nn=30")
        #multiple_features_eda.model_checking(30, scoring='precision')
        print("\n regression_cutoff_50 without resample\nn=30")
        #multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("regression_cutoff_50 logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("regression_cutoff_50 logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')

    def test_algorithms_for_target_tred_cutoff(self):
        print("precision score")
        print("\n\n\n tred_cutoff \n")
        target = "tred_cutoff"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features + self.features_2, target)
        print("\n tred_cutoff with resample\nn=10")
        #multiple_features_eda.model_checking(10, scoring='precision')
        print("\n tred_cutoff without resample\nn=10")
        #multiple_features_eda.model_checking_without_resampling(10, scoring='precision')

        print("\n tred_cutoff with resample\nn=20")
        #multiple_features_eda.model_checking(20, scoring='precision')
        print("\n tred_cutoff without resample\nn=20")
        #multiple_features_eda.model_checking_without_resampling(20, scoring='precision')

        print("\n tred_cutoff with resample\nn=30")
        #multiple_features_eda.model_checking(30, scoring='precision')
        print("\n tred_cutoff without resample\nn=30")
        #multiple_features_eda.model_checking_without_resampling(30, scoring='precision')

        print("tred_cutoff logreg f1 score")
        multiple_features_eda.logistic_regression_grid_search()
        print("tred_cutoff logreg precision score")
        multiple_features_eda.logistic_regression_grid_search(scoring='precision')


    def test_algorithms_for_target_regression(self):
        print("\n\n\n regression_cutoff \n")
        f = ['q6.11_NUMB_pcl2', 'q6.13_SLEEP_pcl1', 'intrusion_pcl2', 'phq2',
                    'q6.1_INTRU_pcl2', 'PCL_Broad1', 'q6.14_ANGER_pcl2',
                    'phq1', 'q6.5_PHYS_pcl1', 'denial2']  # 20

        target = "PCL3"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, f, target)
        multiple_features_eda.regression_model()

        # multiple_features_eda.illigal_genralization_checking(self.X_test, self.y_test)

    # multiple_features_eda.interactions()

    def three_models_combined(self):

        hypertension_features = [
            'trait1', 'trait2', 'lot1', 'PCL1', 'PCL2', 'phq2',
             'active_coping1',  'self_blame1',
              'HL_MAOA', 'HML_FKBP5', 'highschool_diploma', 'T1Acc1n', 'T1Acc1t',
             'q6.15_CONC_pcl1', 'q6.2_DREAM_pcl2',
             'hypertention_pcl1', 'hypertention_pcl2', 'avoidance_pcl2', 'intrusion_pcl1', 'depression_pcl2'
                ]
        avoidance_features =  ['q6.1_INTRU_pcl1', 'q6.2_DREAM_pcl2', 'q6.3_FLASH_pcl2','q6.3_FLASH_pcl1',
             'q6.4_UPSET_pcl1', 'q6.14_ANGER_pcl1', 'q6.7_AVSIT_pcl1', 'q6.7_AVSIT_pcl2',
             'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl2', 'q6.14_ANGER_pcl1',
             'avoidance_pcl1', 'avoidance_pcl2', 'depression_pcl1', 'intrusion_pcl1',
             'PCL_Broad2', 'PCL_Strict1','trait2'
             ]

        intrusion_features = ['trait1', 'q6.5_PHYS_pcl1', 'q6.14_ANGER_pcl2', 'state1', 'PCL1', 'phq1', 'self_distraction1',
                    'hypertention_pcl2', 'venting1', 'PCL2', 'self_distraction2', 'behavioral_disengagement2',
                    'q6.17_STRTL_pcl2', 'substance_use1',  'HML_NPY', 'venting2', 'behavioral_disengagement1',
             'ADHD', 'cd_risc1']
        regression_features = [
            'q6.11_NUMB_pcl2', 'q6.13_SLEEP_pcl1', 'intrusion_pcl2', 'phq2',
            'q6.1_INTRU_pcl2', 'PCL_Broad1', 'q6.14_ANGER_pcl2',
            'phq1', 'q6.5_PHYS_pcl1', 'denial2'
        ]
        depression_features= ['q6.1_INTRU_pcl1', 'q6.2_DREAM_pcl2', 'q6.3_FLASH_pcl2', 'q6.3_FLASH_pcl1',
         'q6.4_UPSET_pcl1', 'q6.14_ANGER_pcl1', 'q6.7_AVSIT_pcl1', 'q6.7_AVSIT_pcl2',
         'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl2', 'q6.14_ANGER_pcl1',
         'avoidance_pcl1', 'avoidance_pcl2', 'depression_pcl1', 'intrusion_pcl1',
         'PCL_Broad2', 'PCL_Strict1', 'trait2'
         ]
        target = "PCL_Strict3"
        multiple_features_eda = EDAMultiFeatureBackend(self.df, self.features, target)
        multiple_features_eda.three_models_combined(intrusion_features, avoidance_features, hypertension_features,
                                                    regression_features, depression_features)


    def target_analysis(self):
        pass

    def make_EDA(self):
        pass

if not mew:
    eda = Model()
    # eda.three_models_combined()
    # eda.three_models_combined()
    #eda.test_algorithms_for_trget_hypertension()
    #eda.test_algorithms_for_trget_avoidance()
    #eda.test_algorithms_for_trget_intrusion()
    #eda.test_algorithms_for_target_depression()
    #eda.test_algorithms_for_target_diagnosis()
    #eda.test_algorithms_for_target_regression_cutoff_33()
    #eda.test_algorithms_for_target_regression_cutoff_50()
    #eda.test_algorithms_for_target_only_avoidance_cutoff()#VX
    #eda.test_algorithms_for_target_tred_cutoff()
    #eda.single_feature_analysis()
    #eda.test_models_for_targets()
    eda.test_models_for_targets()

