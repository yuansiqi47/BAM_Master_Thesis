# Databricks notebook source
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

#df = df.iloc[:,1:]

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

print(log_model.fit().summary().as_latex())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic regression model (Baseline)

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'none', max_iter = 1000)
lr.fit(sm_X_train, sm_y_train)

# COMMAND ----------

cross_val_score(lr, sm_X_train, sm_y_train, scoring = fbeta).mean()

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
        hp.choice('kernel', ['kernel': 'rbf', 
                             {'kernel': 'poly',
                             'degree': hp.quniform('degree', 2, 8, 1)}
                            ]
                 )
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

pd_svm = best_svm.predict_proba(X_test)[:,1]
pd_svm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

algo=tpe.suggest
spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_rf = {
    'type': 'rf',
    'n_estimators' : hp.randint('rf_n_estimators', 1000),
    'max_depth': hp.quniform('rf_max_depth', 1, 20, 1)
    }


# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_rf,
        algo=algo,
        max_evals=100,
        trials=spark_trials)

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space_rf, best_result))


# COMMAND ----------

para_rf = hyperopt.space_eval(search_space_rf, best_result)
del para_rf['type']
best_rf = RandomForestClassifier(**para_rf, random_state = 0)

# COMMAND ----------

#best_rf = RandomForestClassifier(max_depth=20.0, n_estimators=591, random_state=0)
best_rf.fit(sm_X_train, sm_y_train)

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

pd_rf = best_rf.predict_proba(X_test)[:,1]
pd_rf

# COMMAND ----------

# MAGIC %md
# MAGIC ## GB

# COMMAND ----------

algo=tpe.suggest
spark_trials = SparkTrials(parallelism = 32, timeout = 36000)

# COMMAND ----------

search_space_gb = {
    'type': 'gb',
    'n_estimators' : hp.randint('gb_n_estimators', 1000),
    'max_depth': hp.quniform('gb_max_depth', 1, 20, 1),
    'learning_rate': hp.lognormal('learning_rate', 0, 1.0),
    'min_samples_split': hp.uniform('gb_min_samples_split', 0, 0.5),
    'min_samples_leaf': hp.uniform('gb_min_samples_leaf', 0, 0.5)
    }


# COMMAND ----------

with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space_gb,
        algo=algo,
        max_evals=32,
        trials=spark_trials)

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space_gb, best_result))


# COMMAND ----------

para_gb = hyperopt.space_eval(search_space_gb, best_result)
del para_gb['type']
best_gb = GradientBoostingClassifier(**para_gb, random_state = 0)

# COMMAND ----------

best_gb.fit(sm_X_train, sm_y_train)

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

pd_gb = best_gb.predict_proba(X_test)[:,1]
pd_gb

# COMMAND ----------



# COMMAND ----------

df_pd = df.iloc[y_test.index][['gvkey', 'datadate', 'fyear', 'default', 'default_date']].reset_index(drop=True) 
col_list = [y_pred_lr, pd_lr, y_pred_lasso, pd_lasso, y_pred_svm, pd_svm, y_pred_rf, pd_rf, y_pred_gb, pd_gb]

for col in col_list:
    df_pd = pd.concat([df_pd, pd.DataFrame(col)], axis = 1)
    
df_pd.columns = ['gvkey', 'datadate', 'fyear', 'default', 'default_date','pred_lr','pd_lr',
                 'pred_lasso','pd_lasso','pred_svm','pd_svm','pred_rf','pd_rf','pred_gb','pd_gb']
df_pd

# COMMAND ----------

#df_pd.to_csv('/dbfs/FileStore/Siqi thesis/df_pd.csv', index=False)

# COMMAND ----------

#df_pd = pd.read_csv("/dbfs/FileStore/Siqi thesis/df_pd.csv")

# COMMAND ----------

#df_pd['pred_gb'] = y_pred_gb.tolist()
#df_pd['pd_gb'] = pd_gb.tolist()

# COMMAND ----------

#df_pd.to_csv('/dbfs/FileStore/Siqi thesis/df_pd.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Interpretability
# MAGIC Since random forest is the best model so far, we have a look at the model interpretability by getting the global feature importance and local SHAP values
# MAGIC ### feature importance

# COMMAND ----------

importances = best_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)
sorted_indices = np.argsort(importances)[::-1]

# COMMAND ----------

import pandas as pd

forest_importances = pd.Series(importances[sorted_indices], index= sm_X_train.columns[sorted_indices])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std[sorted_indices], ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# COMMAND ----------

import matplotlib.pyplot as plt
 
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP interpretability

# COMMAND ----------

import shap
explainer = shap.TreeExplainer(best_rf)

# COMMAND ----------

# based on all data from test set
shap_values = explainer(X_test)

# COMMAND ----------

# based on all data from test set
shap.plots.beeswarm(shap_values[:,:,0])

# COMMAND ----------

# based on all data from test set
shap.plots.beeswarm(shap_values[:,:,1])

# COMMAND ----------

# global interpretability
# only took a sample
sample = X_test.sample(100)
shap_values_sample = explainer.shap_values(sample)
shap.summary_plot(shap_values_sample, sample, max_display = 20)

# COMMAND ----------

# sample importance 
shap.summary_plot(shap_values_sample[1],sample)

# COMMAND ----------

# local interpretability 
# 1
shap_display = shap.force_plot(explainer.expected_value[1], 
                               shap_values_sample[1][0], 
                               sample.iloc[0], 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------

# local interpretability 
# 2
shap_display = shap.force_plot(explainer.expected_value[1], 
                               shap_values_sample[1][3], 
                               sample.iloc[3], 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------

# local interpretability 
# 3
shap_display = shap.force_plot(explainer.expected_value[1], 
                               shap_values_sample[1][20], 
                               sample.iloc[20], 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------

# local interpretability 
# 4
shap_display = shap.force_plot(explainer.expected_value[1], 
                               shap_values_sample[1][50], 
                               sample.iloc[50], 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------



# COMMAND ----------

import lime
from lime import lime_tabular
 
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(sm_X_train),
    feature_names=sm_X_train.columns,
    class_names=['Non-default','Default'],
    mode='classification'
)

# COMMAND ----------

exp = explainer.explain_instance(
    data_row=X_test.iloc[1], 
    predict_fn = best_rf.predict_proba
)
 
exp.show_in_notebook(show_table=True)

# COMMAND ----------

exp.show_in_notebook(show_table=True).displayHTML()

# COMMAND ----------



# COMMAND ----------


