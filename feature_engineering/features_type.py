


## features groups

features = ["age", "highschool_diploma", "dyslexia", "ADHD",
            "T1Acc1t", "T1Acc1n", "T1bias",
            "phq1", "lot1", "trait1", "state1",
            "PCL1", "PCL_Broad1", "PCL_Strict1",
            "phq2", "lot2", "trait2", "state2",
            "PCL2", "PCL_Broad2", "PCL_Strict2",
            "cd_risc1",
            "active_coping1", "planning1", "positive_reframing1", "acceptance1",
            "humor1", "religion1", "emotional_support1", "instrumental_support1", "self_distraction1", "denial1",
            "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1",
            "active_coping2", "planning2", "positive_reframing2", "acceptance2", "humor2", "religion2",
            "emotional_support2", "instrumental_support2", "self_distraction2", "denial2", "venting2",
            "substance_use2", "behavioral_disengagement2", "self_blame2",
            "trauma_history8_1",
            "HML_5HTT", "HL_MAOA", "HML_NPY", "COMT_Hap1_recode",
            "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5"]

all_t1_features = ["phq1", "lot1", "trait1", "state1", "PCL1", "PCL_Strict1", "active_coping1", "planning1", "positive_reframing1", "acceptance1", "humor1", "religion1", "emotional_support1", "instrumental_support1", "self_distraction1", "denial1", "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1"]
all_t2_features = ["phq2", "lot2", "trait2", "state2", "PCL2", "PCL_Strict2", "active_coping2", "planning2", "positive_reframing2", "acceptance2", "humor2", "religion2", "emotional_support2", "instrumental_support2", "self_distraction2", "denial2", "venting2", "substance_use2", "behavioral_disengagement2", "self_blame2"]

coping_mechanisms_t1 = ["active_coping1", "planning1", "positive_reframing1", "acceptance1",
            "humor1", "religion1", "emotional_support1", "instrumental_support1", "self_distraction1", "denial1",
            "venting1", "substance_use1", "behavioral_disengagement1", "self_blame1"]

coping_mechanisms_t2 = ["active_coping2", "planning2", "positive_reframing2", "acceptance2", "humor2", "religion2",
            "emotional_support2", "instrumental_support2", "self_distraction2", "denial2", "venting2",
            "substance_use2", "behavioral_disengagement2", "self_blame2"]

psych_t1 = ["phq1", "lot1", "trait1", "state1", "PCL1"]
psych_t2 = ["phq2", "lot2", "trait2", "state2", "PCL2"]

genes = ["HML_5HTT", "HL_MAOA", "HML_NPY", "COMT_Hap1_recode",
            "COMT_Hap2_recode", "COMT_Hap1_LvsMH", "HML_FKBP5"]

dot_prob = ["T1Acc1t", "T1Acc1n", "T1bias"]

demographics = ["highschool_diploma", "trauma_history8_1"]

ptsd_t1 = ["PCL1", "PCL_Broad1", "PCL_Strict1", "trauma_history8_1", "cd_risc1"]
ptsd_t2 = ["PCL2", "PCL_Broad2", "PCL_Strict2"]

intrusion_t1 = ['q6.1_INTRU_pcl1', 'q6.2_DREAM_pcl1', 'q6.3_FLASH_pcl1', 'q6.4_UPSET_pcl1', 'q6.5_PHYS_pcl1']
avoidance_t1 = ['q6.6_AVTHT_pcl1', 'q6.7_AVSIT_pcl1', 'q6.8_AMNES_pcl1', 'q6.9_DISINT_pcl1', 'q6.10_DTACH_pcl1',
             'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl1']
hypertension_t1 = ['q6.13_SLEEP_pcl1', 'q6.14_ANGER_pcl1', 'q6.15_CONC_pcl1', 'q6.16_HYPER_pcl1', 'q6.17_STRTL_pcl1']
depression_t1 = ['q6.9_DISINT_pcl1', 'q6.10_DTACH_pcl1', 'q6.11_NUMB_pcl1', 'q6.12_FUTRE_pcl1']

intrusion_t2 = ['q6.1_INTRU_pcl2', 'q6.2_DREAM_pcl2', 'q6.3_FLASH_pcl2', 'q6.4_UPSET_pcl2', 'q6.5_PHYS_pcl2']
avoidance_t2 = ['q6.6_AVTHT_pcl2', 'q6.7_AVSIT_pcl2', 'q6.8_AMNES_pcl2', 'q6.9_DISINT_pcl2', 'q6.10_DTACH_pcl2',
             'q6.11_NUMB_pc21', 'q6.12_FUTRE_pcl2']
hypertension_t2 = ['q6.13_SLEEP_pcl2', 'q6.14_ANGER_pcl2', 'q6.15_CONC_pcl2', 'q6.16_HYPER_pcl2', 'q6.17_STRTL_pcl2']
depression_t2 = ['q6.9_DISINT_pcl2', 'q6.10_DTACH_pcl2', 'q6.11_NUMB_pcl2', 'q6.12_FUTRE_pcl2']


intrusion_t3 = ['q6.1_INTRU_pcl3', 'q6.2_DREAM_pcl3', 'q6.3_FLASH_pcl3', 'q6.4_UPSET_pcl3', 'q6.5_PHYS_pcl3']
avoidance_t3 = ['q6.6_AVTHT_pcl3', 'q6.7_AVSIT_pcl3', 'q6.8_AMNES_pcl3', 'q6.9_DISINT_pcl3', 'q6.10_DTACH_pcl3',
             'q6.11_NUMB_pcl3', 'q6.12_FUTRE_pcl3']

hypertension_t3 = ['q6.13_SLEEP_pcl3', 'q6.14_ANGER_pcl3', 'q6.15_CONC_pcl3', 'q6.16_HYPER_pcl3', 'q6.17_STRTL_pcl3']
depression_t3 = ['q6.9_DISINT_pcl3', 'q6.10_DTACH_pcl3', 'q6.11_NUMB_pcl3', 'q6.12_FUTRE_pcl3']




#________________________________________________________________________


# good features from 1 feature analysis




# good features from 2 features analysis


# good features from delta
#('q6.8_AMNES_pcl1', 'q6.12_FUTRE_pcl1')

# good features from division
# ('trauma_history8_1', 'q6.13_SLEEP_pcl1')
# ('PCL_Broad1', 'q6.8_AMNES_pcl1')
# ('PCL_Strict1', 'PCL_Broad2')
# ('q6.7_AVSIT_pcl1', 'q6.8_AMNES_pcl1')
# ('PCL_Broad2', 'self_blame2')

# good features from sum
# ('q6.4_UPSET_pcl1', 'q6.12_FUTRE_pcl2')
# ('q6.2_DREAM_pcl1', 'q6.12_FUTRE_pcl2')
# ('q6.11_NUMB_pcl1', 'q6.15_CONC_pcl1')
# ('q6.12_FUTRE_pcl1', 'q6.13_SLEEP_pcl2')
# ('q6.5_PHYS_pcl1', 'q6.12_FUTRE_pcl1')
# ('PCL_Strict2', 'q6.12_FUTRE_pcl1')
# ('q6.1_INTRU_pcl2', 'q6.12_FUTRE_pcl2')
# ('q6.13_SLEEP_pcl1', 'intrusion_pcl1')
# ('intrusion_pcl1', 'q6.2_DREAM_pcl2')
# ('behavioral_disengagement1', 'q6.15_CONC_pcl1')
# ('PCL2', 'active_coping1')
# ('PCL_Strict1', 'PCL_Broad2')
# ('q6.7_AVSIT_pcl2', 'q6.9_DISINT_pcl2')
# ('HML_NPY', 'q6.2_DREAM_pcl2')
# ('PCL_Broad2', 'hypertention_pcl1')
# ('phq1', 'emotional_support2')
# ('q6.15_CONC_pcl1', 'q6.12_FUTRE_pcl2')

# good features from multiply
# ('q6.13_SLEEP_pcl1', 'q6.12_FUTRE_pcl2')
# ('phq1', 'q6.5_PHYS_pcl2')
# ('q6.15_CONC_pcl1', 'q6.12_FUTRE_pcl2')
# ('PCL_Broad2', 'self_blame2')
# ('PCL_Broad2', 'positive_reframing2')
# ('q6.4_UPSET_pcl1', 'q6.12_FUTRE_pcl2')
# ('PCL_Broad2', 'q6.9_DISINT_pcl1')
# ('q6.13_SLEEP_pcl1', 'intrusion_pcl1')
# ('avoidance_pcl1', 'hypertention_pcl2')
# ('PCL_Broad2', 'behavioral_disengagement1')
# ('avoidance_pcl1', 'q6.14_ANGER_pcl2')
# ('phq2', 'q6.7_AVSIT_pcl1')
# ('q6.5_PHYS_pcl1', 'q6.12_FUTRE_pcl1')
# ('PCL_Broad2', 'q6.14_ANGER_pcl2')
# ('HML_NPY', 'q6.2_DREAM_pcl2')
# ('PCL_Broad2', 'hypertention_pcl1')
# ('q6.12_FUTRE_pcl1', 'q6.12_FUTRE_pcl2')
# ('phq2', 'q6.15_CONC_pcl1')
# ('PCL_Broad2', 'behavioral_disengagement2')
# ('q6.17_STRTL_pcl1', 'q6.15_CONC_pcl2')


# violance exposure, Arachin, army, war

# relegin. roots,

# values, poitics

# denail phq distraction trait

# active, logic, acceptence, emotional control

#