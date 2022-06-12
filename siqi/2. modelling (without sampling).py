# Databricks notebook source
# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score, make_scorer
from sklearn.impute import KNNImputer
import statsmodels.api as statm
import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt

# COMMAND ----------

# load data
df = pd.read_csv("/dbfs/FileStore/Siqi thesis/df.csv")

# COMMAND ----------

df = df.sample(frac=1,  random_state=42).reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train test split

# COMMAND ----------

X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date'], axis = 1)
y = df['default']

# COMMAND ----------

# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
Xtrans = pd.DataFrame(Xtrans,columns=X.columns)

# COMMAND ----------

Xtrans = pd.get_dummies(Xtrans, columns = ['gsector'], drop_first = True)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(Xtrans, y, stratify = df['default'], test_size = 0.2, random_state=0)

# COMMAND ----------

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# COMMAND ----------

fbeta = make_scorer(fbeta_score, beta=2)
metrics = {'fbeta': fbeta, 'accuracy':'accuracy', 'precision':'precision', 'recall':'recall'}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical logit model

# COMMAND ----------

# logit model on training set
log_model = statm.Logit(y_train, X_train)

# COMMAND ----------

log_model.fit().summary()

# COMMAND ----------

print(log_model.fit().summary().as_latex())

# COMMAND ----------

# MAGIC %md
# MAGIC Creating tables for matrix

# COMMAND ----------

model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
val_scores = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
                          index = model)
evaluation = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
                         index = model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic regression model (Baseline)

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'none', max_iter = 1000)

# COMMAND ----------

accuracy = cross_val_score(lr, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
precision = cross_val_score(lr, X_train, y_train, scoring = 'precision', cv = 5).mean() 
recall = cross_val_score(lr, X_train, y_train, scoring = 'recall', cv = 5).mean()
fbeta_val = cross_val_score(lr, X_train, y_train, scoring = fbeta).mean()

# COMMAND ----------

val_scores.loc['Logistic regression (baseline)', :] = [accuracy, precision, recall, fbeta_val]
val_scores

# COMMAND ----------

lr.fit(X_train, y_train)

# COMMAND ----------

y_pred_lr = lr.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_lr)

# COMMAND ----------

evaluation.loc['Logistic regression (baseline)', :] = [accuracy_score(y_test, y_pred_lr),
                                                       precision_score(y_test, y_pred_lr), 
                                                       recall_score(y_test, y_pred_lr), 
                                                       fbeta_score(y_test, y_pred_lr, beta = 2)]
evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment

# COMMAND ----------

rstate = np.random.default_rng(20220524)
algo=tpe.suggest

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        if params['kernel'] != 'poly':
            del params['degree']
        clf = SVC(**params, random_state = 0)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params, random_state = 0)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params, random_state = 0, penalty = 'l1', max_iter = 3000)
    elif classifier_type == 'gb':
        clf = GradientBoostingClassifier(**params, random_state = 0)
    else:
        return 0
    fbeta_score = cross_val_score(clf, X_train, y_train, scoring = fbeta).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -fbeta_score, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Lasso penalized logistic regression

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
        max_evals=100,
        trials=spark_trials,
        rstate = rstate)

# COMMAND ----------

print(space_eval(search_space_lasso, best_result))

# COMMAND ----------

para_lasso = space_eval(search_space_lasso, best_result)
del para_lasso['type']
best_lasso = LogisticRegression(**para_lasso, random_state = 0, penalty = 'l1', max_iter = 3000)

# COMMAND ----------

accuracy = cross_val_score(best_lasso, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
precision = cross_val_score(best_lasso, X_train, y_train, scoring = 'precision', cv = 5).mean() 
recall = cross_val_score(best_lasso, X_train, y_train, scoring = 'recall', cv = 5).mean()

val_scores.loc['Logistic regression (Lasso)', :] = [accuracy, precision, recall, 
                                                -spark_trials.best_trial['result']['loss']]
val_scores

# COMMAND ----------

best_lasso.fit(X_train, y_train)

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

# MAGIC %md
# MAGIC ## SVM

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_svm = {
    'type': 'svm',
    'C': hp.lognormal('SVM_C', 0, 1.0),
    'kernel': hp.choice('kernel_choice', ['rbf', 'linear','poly']), 
    'degree': hp.quniform('degree', 2, 8, 1)
}

# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_svm,
        algo=algo,
        max_evals=100,
        trials=spark_trials,
        rstate = rstate)

# COMMAND ----------

print(space_eval(search_space_svm, best_result))

# COMMAND ----------

para_svm = space_eval(search_space_svm, best_result)
del para_svm['type']
if para_svm['kernel'] != 'poly':
    del para_svm['degree']
best_svm = SVC(**para_svm, probability = True, random_state = 0)

# COMMAND ----------

accuracy = cross_val_score(best_svm, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
precision = cross_val_score(best_svm, X_train, y_train, scoring = 'precision', cv = 5).mean() 
recall = cross_val_score(best_svm, X_train, y_train, scoring = 'recall', cv = 5).mean()

val_scores.loc['SVM', :] = [accuracy, precision, recall, 
                            -spark_trials.best_trial['result']['loss']]
val_scores

# COMMAND ----------

best_svm.fit(X_train, y_train)

# COMMAND ----------

y_pred_svm = best_svm.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_svm)

# COMMAND ----------

evaluation.loc['SVM', :] = [accuracy_score(y_test, y_pred_svm),
                            precision_score(y_test, y_pred_svm),
                            recall_score(y_test, y_pred_svm), 
                            fbeta_score(y_test, y_pred_svm, beta = 2)]
evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_rf = {
    'type': 'rf',
    'n_estimators' : hp.randint('rf_n_estimators', 2000),
    'max_depth': hp.quniform('rf_max_depth', 1, 10, 1),
    "criterion": hp.choice('cirterion', ['gini', 'entropy', 'log_loss']),
    'min_samples_split': hp.uniform('rf_min_samples_split', 0, 0.2),
    'min_samples_leaf': hp.uniform('rf_min_samples_leaf', 0, 0.1)
    }


# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_rf,
        algo=algo,
        max_evals=100,
        trials=spark_trials,
        rstate = rstate)

# COMMAND ----------

print(space_eval(search_space_rf, best_result))


# COMMAND ----------

para_rf = space_eval(search_space_rf, best_result)
del para_rf['type']
best_rf = RandomForestClassifier(**para_rf, random_state = 0)

# COMMAND ----------

accuracy = cross_val_score(best_rf, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
precision = cross_val_score(best_rf, X_train, y_train, scoring = 'precision', cv = 5).mean() 
recall = cross_val_score(best_rf, X_train, y_train, scoring = 'recall', cv = 5).mean()

val_scores.loc['Random Forest', :] = [accuracy, precision, recall, 
                                      -spark_trials.best_trial['result']['loss']]
val_scores

# COMMAND ----------

best_rf.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf = best_rf.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_rf)

# COMMAND ----------

evaluation.loc['Random Forest', :] = [accuracy_score(y_test, y_pred_rf),
                            precision_score(y_test, y_pred_rf),
                            recall_score(y_test, y_pred_rf), 
                            fbeta_score(y_test, y_pred_rf, beta = 2)]
evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Boosting

# COMMAND ----------

spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_gb = {
    'type': 'gb',
    'n_estimators' : hp.randint('gb_n_estimators', 2000),
    'max_depth': hp.quniform('gb_max_depth', 1, 10, 1),
    'learning_rate': hp.lognormal('learning_rate', 0, 1.0),
    'min_samples_split': hp.uniform('gb_min_samples_split', 0, 0.2),
    'min_samples_leaf': hp.uniform('gb_min_samples_leaf', 0, 0.1)
    }

# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_gb,
        algo=algo,
        max_evals=100,
        trials=spark_trials)

# COMMAND ----------

print(space_eval(search_space_gb, best_result))


# COMMAND ----------

para_gb = space_eval(search_space_gb, best_result)
del para_gb['type']
best_gb = GradientBoostingClassifier(**para_gb, random_state = 0)

# COMMAND ----------

accuracy = cross_val_score(best_gb, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
precision = cross_val_score(best_gb, X_train, y_train, scoring = 'precision', cv = 5).mean() 
recall = cross_val_score(best_gb, X_train, y_train, scoring = 'recall', cv = 5).mean()

val_scores.loc['Gradient Boosting', :] = [accuracy, precision, recall, 
                            -spark_trials.best_trial['result']['loss']]
val_scores

# COMMAND ----------

best_gb.fit(X_train, y_train)

# COMMAND ----------

y_pred_gb = best_gb.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred_gb)

# COMMAND ----------

evaluation.loc['Gradient Boosting', :] = [accuracy_score(y_test, y_pred_gb),
                            precision_score(y_test, y_pred_gb),
                            recall_score(y_test, y_pred_gb), 
                            fbeta_score(y_test, y_pred_gb, beta = 2)]
evaluation

# COMMAND ----------


