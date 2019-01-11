from EDA_backend import EDASingleFeatureBackend, EDAMultiFeatureBackend
import pandas as pd
import xlsxwriter
import os

#.EDASingleFeatureBackend, EDA_backend.EDAMultiFeatureBackend
class EDA:
    original_features = ["age", "highschool_diploma", "Hebrew", "dyslexia", "ADHD", "T1ETBE", "T1Acc1t",
                         "T1Acc1n", "T1bias", "T2Acc1t", "T2Acc1n", "T2bias", "state1", "state2", "trait1",
                         "trait2", "lot1", "lot2", "phq1", "phq2", "PCL1", "PCL2", "cd_risc1", "ptgi2",
                         "active_coping1", "planning1", "positive_reframing1", "acceptance1", "humor1",
                         "religion1", "emotional_support1", "instrumental_support1", "self_distraction1",
                         "denial1", "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1",
                         "active_coping2", "planning2", "positive_reframing2", "acceptance2", "humor2",
                         "religion2", "emotional_support2", "instumental_support2", "self_distraction2",
                         "denial2", "venting2", "substance_use2", "behavioral_disengagement2","self_blam2",
                         "trauma_history8_1", "military_exposure_unit", "HML_5HTT", "HL_MAOA", "HML_NPY",
                         "COMT_Hap1_recode", "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5", "Ashken_scale",
                         "Sephar_scale", "Unknown"]

    target_features = ["PCL3", "PCL_Strict3", "PCL_Broad3"]
    dataset_path = r"../Data/PTSD.xlsx"
    def __init__(self):
        pass

    def preprocessing(self):
        pass

    def single_feature_analysis(self):
        data_for_output_file = []
        for feature in self.original_features:
            eda = EDASingleFeatureBackend(feature)
            eda.plot_distribution()
            eda.analyse_outliers()
            eda.get_unique_values()
            eda.calculate_explained_variance_of_target()
            eda.calculate_parameters_of_weak_clf()
            eda.get_data_sample()
            eda.write_statistic_info()
            data_for_output_file.append(eda.write_results())

        file_path = r"feature_summery"
        file_name = "features summery.xlsx"
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
        df = pd.read_excel(self.dataset_path)
        df = df[self.original_features+self.target_features]
        eda = EDAMultiFeatureBackend(df, include_data_without_target=False)
       # eda.plot_corr_matrix()
        eda.two_features_plots()
        #eda.create_corr_data_file()

    def target_analysis(self):
        pass

    def make_EDA(self):
        pass

eda = EDA()
#eda.single_feature_analysis()
eda.unified_analysis()