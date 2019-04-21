from EDA_backend import EDASingleFeatureBackend, EDAMultiFeatureBackend
import pandas as pd
import xlsxwriter
import os

#.EDASingleFeatureBackend, EDA_backend.EDAMultiFeatureBackend
class EDA:

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
                  'intrusion_pcl1', 'avoidance_pcl1', 'hypertention_pcl1',
                  'q6.1_INTRU_pcl2', 'q6.2_DREAM_pcl2',
                  'q6.3_FLASH_pcl2', 'q6.4_UPSET_pcl2',
                  'q6.5_PHYS_pcl2', 'q6.6_AVTHT_pcl2', 'q6.7_AVSIT_pcl2', 'q6.8_AMNES_pcl2', 'q6.9_DISINT_pcl2',
                  'q6.10_DTACH_pcl2', 'q6.11_NUMB_pcl2', 'q6.12_FUTRE_pcl2', 'q6.13_SLEEP_pcl2',
                  'q6.14_ANGER_pcl2', 'q6.15_CONC_pcl2', 'q6.16_HYPER_pcl2', 'q6.17_STRTL_pcl2',
                  'intrusion_pcl2', 'avoidance_pcl2', 'hypertention_pcl2']
    target_features = ["PCL_Strict3"]
    target_features_2 = ["intrusion", "avoidance", "hypertention"]
    ID = ["ID"]
    dataset_path = r"../Data/PTSD.xlsx"

    def __init__(self):
        path = "C:\‏‏PycharmProjects\PTSD\Data\PTSD.xlsx"
        df = pd.read_excel(path)
        df = df[~df['PCL_Strict3'].isna()]
        df = df[self.features + self.ID + self.target_features]
        df_pcl3 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL3.xlsx")
        df_pcl2 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL2.xlsx")
        df_pcl1 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL1.xlsx")


        df = df.merge(df_pcl1, on="ID", how='outer')
        df = df.merge(df_pcl2, suffixes=('_pcl1', '_pcl2'), on="ID", how='outer')
        df = df.merge(df_pcl3.drop(['PCL3_Strict', 'pcl3', 'PCL3_Broad'], axis=1), on="ID", how='outer')
        df = df[~df['PCL_Strict3'].isna()]
        df.drop(self.ID, inplace=True, axis=1)
        self.df = df

    def single_feature_analysis(self):
        data_for_output_file = []
        for feature in self.features + self.features_2 + self.target_features + self.target_features_2:
            eda = EDASingleFeatureBackend(self.df, feature, self.target_features + self.target_features_2)
            eda.plot_distribution()
            eda.analyse_outliers()
            eda.get_unique_values()
            eda.calculate_explained_variance_of_target()
            eda.calculate_parameters_of_weak_clf()
            eda.get_data_sample()
            eda.write_statistic_info()
            data_for_output_file.append(eda.write_results())

        file_path = r"feature_summery"
        file_name = "features summery mice imputation, outer merge no COMT_Ranked.xlsx"
        workbook = xlsxwriter.Workbook(os.path.join(file_path, file_name))
        worksheet = workbook.add_worksheet()
        row = 0
        for output_data in data_for_output_file:
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



    def unified_analysis(self):

        features = self.features + self.features_2

        multiple_features_eda = EDAMultiFeatureBackend(self.df, features, self.target_features[0])

        #multiple_features_eda.create_corr_data_file()

        #multiple_features_eda.two_features_plots()

        multiple_features_eda.interactions()


    def target_analysis(self):
        pass

    def make_EDA(self):
        pass

eda = EDA()
eda.single_feature_analysis()
#eda.unified_analysis()