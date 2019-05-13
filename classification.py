########### models ################ param
############################################
######## scikit-learn multi-models #########
############################################
import os
import numpy as np
import datetime

import scipy.stats as st


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.externals import joblib
import catboost as cb

from lightgbm.sklearn import LGBMClassifier

def get_regressor(n_jobs = 3,n_iter = 30,n_iter_nt = 3,n_components = 25,cv = 5,
                  seed=42,n_features= 14,is_pca = False,scoring = 'neg_mean_absolute_error'):

    estimator = XGBClassifier(nthreads=-1,tree_method='exact')
    objective = 'binary:logistic'

    # Parameter for XGBoost
    params = {  
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 30),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": st.beta(10, 1),
        "subsample": st.beta(10, 1),
        "gamma": st.uniform(0, 10),
        'objective': [objective],
        'scale_pos_weight': st.randint(0, 2),
        "min_child_weight": st.expon(0, 50),
    #     'lambda': st.uniform(0, 20),
    #     'alpha': st.uniform(0, 20),
    #     'rate_drop':st.uniform(0, 1),
        "seed": [seed],
    }
    xgb = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter, scoring = 'f1') 


        
    #################################################################
    #################################################################

    estimator = LogisticRegression()
    # Parameter for LogisticRegression
    params = {
        "penalty": ['l2'],
        "C": [0.001, 0.01, 0.1, 1, 10],
        "tol": [1e-4, 1e-3, 1e-2, 1e-1],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
        "max_iter": st.randint(50, 100),
        'random_state': [seed],
    } 
    log = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter , scoring = 'f1') 

    #################################################################
    #################################################################

    estimator = KNeighborsClassifier()
    # Parameter for KNeighborsClassifier
    params = {
        "n_neighbors": st.randint(2, 50),
        "weights": ['uniform', 'distance'],
        "algorithm": ['ball_tree', 'kd_tree', 'brute'],
        "leaf_size": st.randint(10, 30),
        "p": st.randint(1, 2),
    }
    knn = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter_nt , scoring = 'f1')        

    #################################################################
    #################################################################

    estimator = RandomForestClassifier()

    # Parameter for RandomForestClassifier
    params = {
        "max_depth": [3, None],
        "max_features": st.randint(1, n_features),
        "min_samples_split": st.randint(2, 10),
        "min_samples_leaf": st.randint(1, n_features),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
        'random_state': [seed],
    }
    rnf = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter, scoring = 'f1')
    
    #################################################################
    #################################################################

    estimator = ExtraTreesClassifier()
    # 
    # Parameter for ExtraTreesClassifier
    params = {
        "n_estimators": st.randint(5, 50),
        "max_depth": [3, None],
        "max_features": st.randint(1, n_features),
        "min_samples_split": st.randint(2, 10),
        "min_samples_leaf": st.randint(1, n_features),
        "bootstrap": [True],
        "oob_score": [True],
        "criterion": ["gini", "entropy"],
        'random_state': [seed],
    }
    ext = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter, scoring = 'f1') 

    #################################################################
    #################################################################

    estimator = AdaBoostClassifier()

    # Parameter for AdaBoost
    params = { 
        'n_estimators':st.randint(10, 100), 
        'learning_rate':st.beta(10, 1), 
        'algorithm':['SAMME', 'SAMME.R'],
        'random_state': [seed],
    }
    ada = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter, scoring = 'f1')         


    #################################################################
    #################################################################
    
    estimator = SVC()

    # Parameter for SVC
    params = {  
        'C':[0.001, 0.01, 0.1, 1, 10], 
        'degree': st.randint(1, 10),
        'shrinking': [True, False],
        'probability': [True],
        'tol': [1e-3],
        'random_state': [seed],
    }
    svc = RandomizedSearchCV(estimator, params, cv=cv,n_jobs=n_jobs, n_iter=n_iter_nt, scoring = 'f1')

    #################################################################
    #################################################################

    params = {'boosting_type': 'gbdt',
              'max_depth' : -1,
              'objective': 'binary',
              'nthread': 3, # Updated from nthread
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class' : 1,
              'metric' : 'binary_error'}

    estimator = LGBMClassifier(boosting_type= 'gbdt',
              objective = 'binary',
              n_jobs = 3, # Updated from 'nthread'
              silent = True,
              max_depth = params['max_depth'],
              max_bin = params['max_bin'],
              subsample_for_bin = params['subsample_for_bin'],
              subsample = params['subsample'],
              subsample_freq = params['subsample_freq'],
              min_split_gain = params['min_split_gain'],
              min_child_weight = params['min_child_weight'],
              min_child_samples = params['min_child_samples'],
              scale_pos_weight = params['scale_pos_weight'])




    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [40,30],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt'],
        'objective' : ['binary'],
        'random_state' : [42,502], # Updated from 'seed'
        'colsample_bytree' : [0.65, 0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
        }

    params = {  
                "boosting_type": ["gbdt","rf","dart"],
                "colsample_bytree": st.beta(10, 1),
                "learning_rate": st.uniform(0.05, 0.4),
                "max_depth": st.randint(3, 30),
                "min_child_weight": st.expon(0, 50),
                "n_estimators": st.randint(3, 40),
                "num_leaves": st.randint(30, 50),
                'objective': ['binary'],
                "subsample": st.beta(10, 1),
                "seed": [seed],
            }

    lgb = GridSearchCV(estimator, gridParams, verbose=0, cv=cv,n_jobs=n_jobs , scoring = 'f1')

    #################################################################
    #################################################################

    elas = ElasticNetCV(cv=5, random_state=0,l1_ratio=[0.05,.1,0.15,.2, .5, .9,0.95, 1],n_jobs=-1
                                 ,normalize=False, positive=False)
    bag = BaggingRegressor(elas,n_jobs = n_jobs, n_estimators=30,warm_start =True,bootstrap_features =True)
    return  [("xbg", xgb),
             ("knn", knn),
             ("rnf", rnf),
             ("ext", ext),
             ("ada", ada),
           #  ("svc", svc),
           #  ("bag", bag),
             ("elas", elas)]

