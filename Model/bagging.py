import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer






class BaggingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model=RandomForestClassifier(n_estimators=100)):
        self.model = model
        self.voting_classifiers = []

    def fit(self, X, y):

        negative_count = len(y) - sum(y)
        positive_count = sum(y)
        rounds = negative_count // positive_count

        left = negative_count - (positive_count * (negative_count // positive_count))

        X, y = self.unison_shuffled_copies(X.values, y.values)

        negative = X[y == 0]
        positive = X[y == 1]

        negative_index = 0

        for i in range(int(rounds)):
            if i < left:
                delta = int(positive_count + 1)
                Xs = np.concatenate((negative[negative_index: negative_index + delta], positive))
                negative_index += delta
                Ys = np.concatenate((np.zeros(delta), np.ones(delta - 1)))
            else:
                delta = int(positive_count)
                Xs = np.concatenate((negative[negative_index: negative_index + delta], positive))
                negative_index += delta
                Ys = np.concatenate((np.zeros(delta), np.ones(delta)))

            Xs, Ys = self.unison_shuffled_copies(Xs, Ys)
            self.voting_classifiers.append(self.model.fit(Xs, Ys))

    def predict(self, X):
        amount = np.zeros(X.shape[0])
        for model in self.voting_classifiers:
            amount += model.predict(X)

        amount = amount >= (len(self.voting_classifiers)//2)
        return amount



    def get_params(self, deep = False):
        return {'model':self.model}

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]





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

features_0 = ['q6.11_NUMB_pcl2', 'q6.13_SLEEP_pcl1',
            'intrusion_pcl2', 'phq2', 'q6.1_INTRU_pcl2', 'PCL_Broad1', 'q6.14_ANGER_pcl2',
            'phq1', 'q6.5_PHYS_pcl1', 'denial2', 'avoidance_pcl2', 'positive_reframing1',
            'intrusion_pcl1', 'q6.5_PHYS_pcl2', 'q6.13_SLEEP_pcl2',
            'q6.3_FLASH_pcl2']
path = "C:\‏‏PycharmProjects\PTSD\Data\PTSD.xlsx"
df = pd.read_excel(path)
df = df[~df['PCL_Strict3'].isna()]
df = df[["ID", 'PCL_Strict3']]
df_pcl3 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL3.xlsx")
df_pcl2 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL2.xlsx")
df_pcl1 = pd.read_excel("C:\‏‏PycharmProjects\PTSD\Data\questionnaire6PCL1.xlsx")

df = df.merge(df_pcl1, on="ID")
df = df.merge(df_pcl2, suffixes=('_pcl1', '_pcl2'), on="ID")
df = df.merge(df_pcl3.drop(['PCL3_Strict', 'pcl3', 'PCL3_Broad'], axis=1), on="ID")
df = df[features_0 + ['PCL_Strict3']].dropna()
X = df[features_0]
Y = df['PCL_Strict3']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=271828, stratify=Y)
scores = cross_val_score(BaggingClassifier(), X_train, y_train, scoring='f1', cv=10)
print(sum(scores) / len(scores))