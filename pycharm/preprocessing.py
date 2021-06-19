from scipy.stats import zscore
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import json
#rom fancyimpute import IterativeImputer

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def calculated_correlated(X_train, threshold=0.75):
    pear_feats = []

    corr = X_train.corr()
    for ft in X_train.columns:
        add = True
        for j in pear_feats:
            if np.abs(corr[ft][j]) > threshold:
                add = False
        if add:
            pear_feats.append(ft)

    spear_feats = []
    corr = X_train[pear_feats].corr("spearman")
    for ft in pear_feats:
        add = True
        for j in spear_feats:
            if np.abs(corr[ft][j]) > threshold:
                add = False
        if add:
            spear_feats.append(ft)

    kendall_feats = []
    corr = X_train[spear_feats].corr("kendall")
    for ft in spear_feats:
        add = True
        for j in kendall_feats:
            if np.abs(corr[ft][j]) > threshold:
                add = False
        if add:
            kendall_feats.append(ft)

    return kendall_feats


def removal_correlated(X_train, X_test=None):
    feats = calculated_correlated(X_train)
    if X_test is None:
        return X_train[feats]
    else:
        return X_train[feats], X_test[feats]


def PHQ_calculator(df):
    variables_path = r"variables.json"
    with open(variables_path) as f:
        variables = json.load(f)

    params = ['phq_physical', 'phq_feels', 'phq_an', 'phq_func', 'phq_symptomatic', 'phq_threshold']

    phq_physical, phq_feels, phq_an, phq_func, phq_symptomatic, phq_threshold = [variables[param] for param in params]


    df['phq_physical_score'] = (df[phq_physical]).sum(axis=1)
    df['binary_phq_physical'] = ((df[phq_physical] > phq_symptomatic).sum(axis=1) > 0).astype(int)

    df['phq_feels_score'] = (df[phq_feels]).sum(axis=1)
    df['binary_phq_feels'] = ((df[phq_feels] > phq_symptomatic).sum(axis=1) > 0).astype(int)

    df['phq_an_score'] = (df[phq_an]).sum(axis=1)
    df['binary_phq_an'] = ((df[phq_an] > phq_symptomatic).sum(axis=1) > 0).astype(int)

    df['phq_func_score'] = (df[phq_func]).sum(axis=1)
    df['binary_phq_func'] = ((df[phq_func] > phq_symptomatic).sum(axis=1) > 0).astype(int)

    phqs = [f"T1q5.{i}" for i in range(1,10)]
    df['phq_score'] = (df[phqs]).sum(axis=1)
    df['phq_mean'] = (df[phqs]).mean(axis=1)
    df['phq_std'] = (df[phqs]).std(axis=1)

    df['binary_phq'] = ((df[phqs] > phq_symptomatic).sum(axis=1) > 2).astype(int)

    return df

def PCL_calculator(df):
    # import predefined variables
    variables_path = r"variables.json"
    with open(variables_path) as f:
        variables = json.load(f)

    params = ['intrusion', 'tred', 'avoidance', 'hyper', 'symptomatic_cutoff', 'intrusion_cutoff', 'tred_cutoff', 'avoidance_cutoff', 'hyper_cutoff']

    intrusion, tred, avoidance, hyper, symptomatic_cutoff, intrusion_cutoff, tred_cutoff, avoidance_cutoff, hyper_cutoff = [variables[param] for param in params]

    df['intrusion_score'] = (df[intrusion]).sum(axis=1)
    df['intrusion_mean'] = (df[intrusion]).mean(axis=1)
    df['intrusion_std'] = (df[intrusion]).std(axis=1)

    df['binary_intrusion'] = ((df[intrusion] > symptomatic_cutoff).sum(axis=1) > intrusion_cutoff).astype(int)

    df['tred_score'] = (df[tred]).sum(axis=1)
    df['tred_mean'] = (df[tred]).mean(axis=1)
    df['tred_std'] = (df[tred]).std(axis=1)
    df['binary_tred'] = ((df[tred] > symptomatic_cutoff).sum(axis=1) > tred_cutoff).astype(int)
    ######################################

    df['avoidance_score'] = (df[avoidance]).sum(axis=1)
    df['avoidance_mean'] = (df[avoidance]).mean(axis=1)
    df['avoidance_std'] = (df[avoidance]).std(axis=1)
    df['binary_avoidance'] = ((df[avoidance] > symptomatic_cutoff).sum(axis=1) > avoidance_cutoff).astype(int)


    df['hyper_score'] = (df[hyper]).sum(axis=1)
    df['hyper_mean'] = (df[hyper]).mean(axis=1)
    df['hyper_std'] = (df[hyper]).std(axis=1)
    df['binary_hyper'] = ((df[hyper] > symptomatic_cutoff).sum(axis=1) > hyper_cutoff).astype(int)


    
    return df


def clean_name(X):
    rename_cols = {i: i.replace('.', '').replace(' ', '') for i in X.columns}
    return X.rename(rename_cols, axis=1)


def stds(X):
    variables_path = r"variables.json"
    
    with open(variables_path) as f:
        variables = json.load(f)
    
    params = ['t1_features']
    t1_features = [variables[param] for param in params]
    for i in t1_features[0]:
        col = clean_name(X[[i]])
        X[f"z_score_{col.columns[0]}"] = zscore(col)

    return X


def stats(X_train, X_test=None):

    if X_test is not None:
        m = ols('PCL_Strict1 ~ phq1  + T1Acc1t + T1Acc1n + T1bias', X_test.append(X_train, ignore_index=True)).fit()
        infl = m.get_influence()
        sm_fr = infl.summary_frame()
        X_test[['cooks_d_PCL_score', 'dffits_PCL_score', 'standard_resid_PCL_score']] = sm_fr[['cooks_d',
                                                                                                              'dffits',
                                                                                                              'standard_resid']][
                                                                                                       :X_test.shape[
                                                                                                           0]:]

        return X_test


    else:
        m = ols('PCL_Strict1 ~ phq1 + T1Acc1t + T1Acc1n + T1bias', X_train).fit()
        infl = m.get_influence()
        sm_fr = infl.summary_frame()

        X_train[['cooks_d_PCL_score', 'dffits_PCL_score', 'standard_resid_PCL_score']] = sm_fr[
            ['cooks_d', 'dffits', 'standard_resid']]
        return X_train


def outliers(X_train, X_test=None, features=None, name=''):

    for i, model in enumerate([IsolationForest, EllipticEnvelope, OneClassSVM, LocalOutlierFactor]):

        model = IsolationForest(contamination=0.1)
        if features:
            pred = model.fit_predict(X_train[features])
        else:
            pred = model.fit_predict(X_train)

        X_train[f"{name}_outlier_detection_{i}"] = pred

        if X_test is not None:
            if features:
                pred = model.predict(X_test[features])
            else:
                pred = model.predict(X_test)

            X_test[f"{name}_outlier_detection_{i}"] = pred

    if X_test is not None:
        return X_train, X_test
    else:
        return X_train


def cv_preprocessing(X_train, X_test=None, random_state=None):
    variables_path = r"variables.json"
    with open(variables_path) as f:
        variables = json.load(f)
        t1_features, cogni = variables['t1_features'], variables['cogni']
        pcl = variables['questionnaires']['PCL'][:17]

    mice = KNNImputer()
    columns = X_train.columns
    X_train = pd.DataFrame(mice.fit_transform(X_train), columns=columns)

    #X_train = stds(X_train)
    #X_train = stats(X_train)
    #X_train = removal_correlated(X_train)
    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    # X_train = pd.DataFrame(ss.fit_transform(X_train), columns=columns)
    if X_test is not None:
        X_test = pd.DataFrame(mice.transform(X_test), columns=columns)
        #X_test = stds(X_test)
        #X_test = stats(X_train, X_test)
        #_, X_test = removal_correlated(X_train, X_test)
        # X_test = ss.transform(X_test)
        # X_test = pd.DataFrame(ss.transform(X_test), columns=columns)

        X_train, X_test = outliers(X_train, X_test, features=[f"T1q5.{i}" for i in range(1, 10)], name='phq9')
        #X_train, X_test = outliers(X_train, X_test, features=pcl, name='PCL')
        X_train, X_test = outliers(X_train, X_test, features=cogni, name='cogni')
        X_train, X_test = outliers(X_train, X_test, features=t1_features, name='t1')

        return X_train, X_test
    else:
        return X_train