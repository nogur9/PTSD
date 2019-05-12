

import pandas as pd
import numpy as np
#### t1_t2 delta


### useful features from excel interactions

### variance between symptoms

### stait trait phq9 cutoffs

### pcl high corr features

class FeatureEngineering:
    pass

    def __init__(self, df, target):
        self.df = df
        self.out_df = pd.DataFrame()
        self.target = target

    def engineer_features(self):
        if self.target == "intrusion_cutoff":
            self.engineer_intrusion()

        elif self.target == "avoidance_cutoff":
            self.engineer_avoidance()

        elif self.target == "hypertention_cutoff":
            self.engineer_hypertension()

        elif self.target == "depression_cutoff":
            self.engineer_depression()

        elif self.target == "diagnosis":
            self.engineer_pcl_strict_3()

        elif self.target == "only_avoidance_cutoff":
            self.engineer_only_avoidance()

        elif self.target == "regression_cutoff_33":
            self.engineer_regression_cutoff_33()

        elif self.target == "regression_cutoff_50":
            self.engineer_regression_cutoff_50()

        elif self.target == "tred_cutoff":
            self.engineer_tred_cutoff()

        self.df = self.df.join(self.out_df, how="outer")

        return self.df

    def engineer_hypertension(self):
        delta = [('highschool_diploma', 'hypertention_pcl1'), ('T1Acc1t', 'hypertention_pcl2'),
                 ('trait1', 'q6.13_SLEEP_pcl1')]
        division = [('PCL1', 'COMT_Hap1_recode')]
        multiply = [('PCL1', 'state2'), ('PCL1', 'trait2'), ('state1', 'PCL2')]
        summation = [('q6.15_CONC_pcl1', 'hypertention_pcl2'), ('intrusion_pcl1', 'hypertention_pcl1'),
                     ('q6.15_CONC_pcl1', 'depression_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp
        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_avoidance(self):
        delta = [('PCL_Strict1', 'q6.7_AVSIT_pcl2')]
        division = []
        multiply = [('avoidance_pcl1', 'q6.7_AVSIT_pcl2'), ('avoidance_pcl1', 'avoidance_pcl2'),
                    ('avoidance_pcl1', 'q6.12_FUTRE_pcl2'), ('trait2', 'avoidance_pcl2'),
                    ('q6.4_UPSET_pcl1', 'avoidance_pcl2'), ('q6.3_FLASH_pcl1', 'avoidance_pcl2'),
                    ('q6.3_FLASH_pcl1', 'q6.12_FUTRE_pcl2'), ('q6.5_PHYS_pcl1', 'avoidance_pcl2')]
        summation = [('avoidance_pcl1', 'q6.7_AVSIT_pcl2'), ('avoidance_pcl1', 'q6.12_FUTRE_pcl2'),
                    ('PCL_Broad1', 'q6.7_AVSIT_pcl2'), ('q6.7_AVSIT_pcl1', 'q6.12_FUTRE_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            print(sum((tmp == np.inf) | (tmp == -np.inf)))
            tmp.replace([np.inf, -np.inf], 0)
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_intrusion(self):
        delta = [('trait1', 'PCL_Broad1'), ('PCL1', 'HML_NPY'), ('PCL1', 'q6.17_STRTL_pcl1')]
        division = []#[('PCL1', 'cd_risc1')]
        multiply = [('state1', 'PCL1')]
        summation = [('PCL1', 'venting2'), ('phq1', 'behavioral_disengagement1'),
                    ('state1', 'PCL1'), ('intrusion_pcl1', 'hypertention_pcl1')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            print(sum((tmp == np.inf) | (tmp == -np.inf)))
            tmp.replace([np.inf, -np.inf], 0.0)
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp


    def engineer_only_avoidance(self):
        delta = [('PCL1', 'q6.11_NUMB_pcl2')]
        division = []
        multiply = [('PCL1', 'trait2'), ('self_distraction1', 'q6.6_AVTHT_pcl2'),
                    ('PCL2', 'q6.4_UPSET_pcl1'), ('phq1', 'cd_risc1'), ('state1', 'q6.7_AVSIT_pcl2')]
        summation = [('q6.6_AVTHT_pcl1', 'hypertention_pcl2'), ('q6.4_UPSET_pcl1', 'hypertention_pcl2'),
                     ('avoidance_pcl1', 'hypertention_pcl2'), ('PCL1', 'HL_MAOA'), ('self_blame1', 'q6.4_UPSET_pcl1'),
                     ('phq1', 'avoidance_pcl1')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_depression(self):
        delta = [('PCL_Broad1', 'depression_pcl2'), ('q6.8_AMNES_pcl1', 'depression_pcl2')]
        division = []# [('trauma_history8_1', 'avoidance_pcl1'), ('PCL_Broad1', 'depression_pcl2')]
        multiply = [('q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl2')]
        summation = [('q6.3_FLASH_pcl1', 'avoidance_pcl2'), ('q6.10_DTACH_pcl1', 'depression_pcl2'),
                     ('avoidance_pcl1', 'q6.7_AVSIT_pcl2'), ('q6.10_DTACH_pcl1', 'avoidance_pcl2'),
                     ('q6.11_NUMB_pcl1', 'avoidance_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_regression_cutoff_33(self):
        delta = [('phq2', 'acceptance1'), ('highschool_diploma', 'q6.12_FUTRE_pcl2'),
                 ('phq2', 'HL_MAOA'), ('lot1', 'PCL2')]
        division = [] # [('emotional_support1', 'hypertention_pcl2'),
                    #('highschool_diploma', 'q6.12_FUTRE_pcl2'), ('COMT_Hap2_recode', 'hypertention_pcl2')]
        multiply = [('q6.11_NUMB_pcl1', 'hypertention_pcl2'), ('phq2', 'hypertention_pcl1'),
                    ('q6.14_ANGER_pcl1', 'q6.15_CONC_pcl2'), ('PCL2', 'hypertention_pcl1'),
                    ('q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl2'),('q6.11_NUMB_pcl1', 'q6.15_CONC_pcl2'),
                    ('q6.15_CONC_pcl1', 'q6.12_FUTRE_pcl2'), ('q6.10_DTACH_pcl1', 'hypertention_pcl2'),
                    ('trait1', 'phq2')]
        summation = [('q6.11_NUMB_pcl1', 'q6.15_CONC_pcl2'), ('q6.1_INTRU_pcl1', 'depression_pcl2'),
                     ('q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_regression_cutoff_50(self):
        delta = [('T1Acc1t', 'PCL_Broad2'), ('PCL_Strict1', 'intrusion_pcl2')]
        division = []# [('PCL_Strict1', 'PCL_Strict2')]
        multiply = [('HML_5HTT', 'intrusion_pcl2'), ('substance_use2', 'q6.14_ANGER_pcl2')]
        summation = [('depression_pcl1', 'intrusion_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_pcl_strict_3(self):
        delta = [('PCL_Strict1', 'intrusion_pcl2')]
        division = [] #[('PCL_Broad1', 'q6.8_AMNES_pcl2'), ('PCL_Strict1', 'PCL_Strict2')]
        multiply = [('q6.15_CONC_pcl1', 'q6.12_FUTRE_pcl2'), ('HML_5HTT', 'intrusion_pcl2')]
        summation = [('depression_pcl1', 'intrusion_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

    def engineer_tred_cutoff(self):
        delta = [('PCL_Strict1', 'intrusion_pcl2')]
        division = []
        multiply = [('PCL_Broad1', 'q6.8_AMNES_pcl2'), ('PCL_Strict1', 'PCL_Strict2'),
                    ('q6.15_CONC_pcl1', 'q6.12_FUTRE_pcl2'), ('HML_5HTT', 'intrusion_pcl2')]
        summation = [('depression_pcl1', 'intrusion_pcl2')]

        for (i, j) in delta:
            tmp = self.df[i] - self.df[j]
            self.out_df[f"delta{i} {j}"] = tmp

        for (i, j) in division:
            tmp = self.df[i] / self.df[j]
            self.out_df[f"division{i} {j}"] = tmp

        for (i, j) in multiply:
            tmp = self.df[i] * self.df[j]
            self.out_df[f"multiply{i} {j}"] = tmp

        for (i, j) in summation:
            tmp = self.df[i] + self.df[j]
            self.out_df[f"sum{i} {j}"] = tmp

