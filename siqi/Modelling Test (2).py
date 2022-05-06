# Databricks notebook source
pip install imbalanced-learn

# COMMAND ----------

# Import packages
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import statsmodels.api as statm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score, make_scorer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

# COMMAND ----------

# load data
df = pd.read_csv("/dbfs/FileStore/Siqi thesis/df.csv")

# COMMAND ----------

df = df.drop(columns = ['P2'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train test split

# COMMAND ----------

df = df.iloc[:,1:]

# COMMAND ----------

# X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date', 'gsector'], axis = 1)
X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date'], axis = 1)
y = df['default']

# COMMAND ----------

# knn impute
from sklearn.impute import KNNImputer
# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
Xtrans = pd.DataFrame(Xtrans,columns=X.columns)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(Xtrans, y, stratify = df['default'], test_size = 0.3, random_state=0)

# COMMAND ----------

# smote
sm = SMOTE(random_state=0)
columns = X_train.columns
sm_X_train, sm_y_train = sm.fit_resample(X_train, y_train)
sm_X_train = pd.DataFrame(data=sm_X_train,columns=columns )
sm_y_train = pd.DataFrame(data=sm_y_train,columns=['default'])
# check the default proportion
print("Proportion of default data in oversampled data is ",len(sm_y_train[sm_y_train['default']==1])/len(sm_X_train))

# COMMAND ----------

sm_X_train = sm_X_train.sample(frac=1,  random_state=42).reset_index(drop=True)
sm_y_train = sm_y_train.sample(frac=1,  random_state=42).reset_index(drop=True)

# COMMAND ----------

sm_y_train = np.asarray(sm_y_train['default'])

# COMMAND ----------

fbeta = make_scorer(fbeta_score, beta=2)
metrics = {'fbeta': fbeta, 'accuracy':'accuracy', 'precision':'precision', 'recall':'recall'}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical logit model

# COMMAND ----------

# logit model on training set
log_model = statm.Logit(sm_y_train, sm_X_train)

# COMMAND ----------

log_model.fit().summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic regression model (Baseline)

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'none', max_iter = 1000)
lr.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_lr = lr.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_lr)

# COMMAND ----------

model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
evaluation = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
                         index = model)

# COMMAND ----------

evaluation.loc['Logistic regression (baseline)', :] = [accuracy_score(y_test, y_pred_lr),
                                                       precision_score(y_test, y_pred_lr), 
                                                       recall_score(y_test, y_pred_lr), 
                                                       fbeta_score(y_test, y_pred_lr, beta = 2)]
evaluation

# COMMAND ----------

pd_lr = lr.predict_proba(X_test)[:, 1]
pd_lr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import mlflow

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params, random_state = 0)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params, random_state = 0)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params, random_state = 0, penalty = 'l1')
    elif classifier_type == 'gb':
        clf = GradientBoostingClassifier(**params, random_state = 0)
    else:
        return 0
    fbeta_score = cross_val_score(clf, sm_X_train, sm_y_train, scoring = fbeta).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -fbeta_score, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Lasso penalized logistic regression

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32)

# COMMAND ----------

search_space_lasso = {
        'type': 'logreg',
        'C': hp.lognormal('LR_C', 0, 1.0),
        'solver': hp.choice('solver', ['liblinear', 'saga'])
    }


# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_lasso,
        algo=algo,
        max_evals=32,
        trials=spark_trials)

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space_lasso, best_result))

# COMMAND ----------

para_lasso = hyperopt.space_eval(search_space_lasso, best_result)
del para_lasso['type']
best_lasso = LogisticRegression(**para_lasso, random_state = 0, penalty = 'l1')
best_lasso.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_lasso = best_lasso.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_lasso)

# COMMAND ----------

evaluation.loc['Logistic regression (Lasso)', :] = [accuracy_score(y_test, y_pred_lasso),
                                                       precision_score(y_test, y_pred_lasso), 
                                                       recall_score(y_test, y_pred_lasso), 
                                                       fbeta_score(y_test, y_pred_lasso, beta = 2)]
evaluation

# COMMAND ----------

pd_lasso = best_lasso.predict_proba(X_test)[:,1]
pd_lasso

# COMMAND ----------

# MAGIC %md
# MAGIC ## SVM

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_svm = {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['poly', 'rbf']),
        'degree': hp.quniform('degree', 2, 8, 1)
    }

# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_svm,
        algo=algo,
        max_evals=32,
        trials=spark_trials)

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space_svm, best_result))


# COMMAND ----------

para_svm = hyperopt.space_eval(search_space_svm, best_result)
del para_svm['type']
best_svm = SVC(**para_svm, probability = True, random_state = 0)

# COMMAND ----------

best_svm.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred = best_svm.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)

# COMMAND ----------

evaluation.loc['SVM', :] = [accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred),
                            recall_score(y_test, y_pred), 
                            fbeta_score(y_test, y_pred, beta = 2)]
evaluation

# COMMAND ----------

pd_svm = best_svm.predict_proba(X_test)[:,1]
pd_svm

# COMMAND ----------

# MAGIC %md
# MAGIC ## GB

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import mlflow

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params, random_state = 0)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params, random_state = 0)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params, random_state = 0, penalty = 'l1')
    elif classifier_type == 'gb':
        clf = GradientBoostingClassifier(**params, random_state = 0)
    else:
        return 0
    fbeta_score = cross_val_score(clf, sm_X_train, sm_y_train, scoring = fbeta).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -fbeta_score, 'status': STATUS_OK}

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32)

# COMMAND ----------

search_space_gb = {
        'type': 'gb',
        'n_estimators' : hp.quniform('GB_n_estimators', 50, 500, 50),
#        'max_depth': hp.quniform('GB_max_depth', 4, 20, 4),
 #       'learning_rate': hp.lognormal('learning_rate', 0, 1.0)
    }


# COMMAND ----------

with mlflow.start_run():
#    mlflow.log_metric("fbeta", fbeta_score)
    best_result = fmin(
        fn=objective, 
        space=search_space_gb,
        algo=algo,
        max_evals=10,
        trials=spark_trials)

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space_gb, best_result))


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


