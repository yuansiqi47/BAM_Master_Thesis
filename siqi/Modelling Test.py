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
import mlflow

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

lr = LogisticRegression(random_state=0, penalty = 'none')
lr.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred = lr.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)

# COMMAND ----------

model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
evaluation = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
                         index = model)

# COMMAND ----------

evaluation.loc['Logistic regression (baseline)', :] = [accuracy_score(y_test, y_pred),
                                                       precision_score(y_test, y_pred), 
                                                       recall_score(y_test, y_pred), 
                                                       fbeta_score(y_test, y_pred, beta = 2)]
evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ##Experiment

# COMMAND ----------

mlflow.autolog(max_tuning_runs=None)
    
parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'max_depth': [5, 10, 15, 20]}
rf = RandomForestClassifier(random_state=0)
    
with mlflow.start_run(run_name='rf_grid_search') as run:
    clf = GridSearchCV(rf, parameters, cv = 5, scoring = metrics, refit = 'fbeta')
    clf.fit(sm_X_train, sm_y_train)


# COMMAND ----------

best_rf = clf.best_estimator_
best_rf.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_rf = best_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))

evaluation.loc['SVM', :] = [accuracy_score(y_test, y_pred_rf),
                            precision_score(y_test, y_pred_rf),
                            recall_score(y_test, y_pred_rf), 
                            fbeta_score(y_test, y_pred_rf, beta = 2)]
print(evaluation)

pd_rf = best_rf.predict_proba(X_test)[:,1]
print(pd_rf)

# COMMAND ----------

mlflow.autolog(max_tuning_runs=None)
    
parameters = {'n_estimators' : [50, 100, 150, 200, 250, 1000],
              'max_depth': [2, 4, 6, 8],
             'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5]}
gb = GradientBoostingClassifier(random_state=0)
    
with mlflow.start_run(run_name='gb_grid_search') as run:
    clf = GridSearchCV(gb, parameters, cv = 5, scoring = metrics, refit = 'fbeta')
    clf.fit(sm_X_train, sm_y_train)

# COMMAND ----------

best_gb = clf.best_estimator_
best_gb.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_gb = best_gb.predict(X_test)

print(confusion_matrix(y_test, y_pred_gb))

evaluation.loc['SVM', :] = [accuracy_score(y_test, y_pred_gb),
                            precision_score(y_test, y_pred_gb),
                            recall_score(y_test, y_pred_gb), 
                            fbeta_score(y_test, y_pred_gb, beta = 2)]
print(evaluation)

pd_gb = best_gb.predict_proba(X_test)[:,1]
print(pd_gb)

# COMMAND ----------

mlflow.autolog(max_tuning_runs=None)
    
parameters = {'c' : [0.5, 1, 5, 10],
              'kernel': ['rbf', 'poly'],
              'degree': [3, 4, 5, 6]}

svm = SVC(random_state=0)
    
with mlflow.start_run(run_name='svm_grid_search') as run:
    clf = GridSearchCV(svm, parameters, cv = 5, scoring = metrics, refit = 'fbeta')
    clf.fit(sm_X_train, sm_y_train)


# COMMAND ----------

best_svm = clf.best_estimator_
best_svm.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_svm = best_svm.predict(X_test)

print(confusion_matrix(y_test, y_pred_svm))

evaluation.loc['SVM', :] = [accuracy_score(y_test, y_pred_svm),
                            precision_score(y_test, y_pred_svm),
                            recall_score(y_test, y_pred_svm), 
                            fbeta_score(y_test, y_pred_svm, beta = 2)]
print(evaluation)

pd_svm = best_svm.predict_proba(X_test)[:,1]
print(pd_svm)

# COMMAND ----------

mlflow.autolog(max_tuning_runs=None)
    
parameters = {'C' : [0.01, 0.1, 0.5, 1, 10, 20]}

lasso = LogisticRegression(random_state=0, penalty = 'l1', solver = 'liblinear', max_iter=1000)
    
with mlflow.start_run(run_name='lasso_grid_search') as run:
    clf = GridSearchCV(lasso, parameters, scoring = metrics, refit = 'fbeta')
    clf.fit(sm_X_train, sm_y_train)


# COMMAND ----------

best_lasso = SVC(**clf.best_params, probability = True)
best_lasso.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred_lasso = best_lasso.predict(X_test)

print(confusion_matrix(y_test, y_pred_lasso))

evaluation.loc['Logistic regression (Lasso)', :] = [accuracy_score(y_test, y_pred_lasso),
                                                    precision_score(y_test, y_pred_lasso),
                                                    recall_score(y_test, y_pred_lasso), 
                                                    fbeta_score(y_test, y_pred_lasso, beta = 2)]
print(evaluation)

pd_lasso = best_lasso.predict_proba(X_test)[:,1]
print(pd_lasso)

# COMMAND ----------

# MAGIC %md
# MAGIC ## experiment with cross_val_score

# COMMAND ----------

import mlflow
from sklearn.model_selection import cross_val_score

# COMMAND ----------

### gradient boosting
for n in [50, 100, 150, 200, 250, 1000]:
    for d in [2, 4, 6, 8]:
        for l in [0.01, 0.05, 0.1, 0.25, 0.5]:
            with mlflow.start_run(run_name='GB_cv') as run:
                model = GradientBoostingClassifier(random_state=0, n_estimators = n, max_depth = d, learning_rate = l)
                #model.fit(sm_X_train, sm_y_train)
                #y_pred = model.predict(X_test)
                mlflow.log_param('n_estimators', n)
                mlflow.log_param('max_depth', d)
                mlflow.log_param('learning_rate', l)
                
                fbeta_score = cross_val_score(model, sm_X_train, sm_y_train, scoring = fbeta, cv = 5).mean()
                accuracy = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'accuracy', cv = 5).mean()
                precision = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'precision', cv = 5).mean() 
                recall = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'recall', cv = 5).mean()
                    
               # accuracy = accuracy_score(y_test, y_pred)
                #precision = precision_score(y_test, y_pred)
                #recall = recall_score(y_test, y_pred)
                #fbeta = fbeta_score(y_test, y_pred, beta = 2)
                    
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("fbeta", fbeta_score)
                    


# COMMAND ----------

### Random Forest
for n in [100, 200, 300, 500, 1000]:
    for d in [4, 6, 8, 15, 30]:
        with mlflow.start_run(run_name='RF') as run:
            model = RandomForestClassifier(random_state=0, n_estimators = n, max_depth = d)
            #model.fit(sm_X_train, sm_y_train)
            #y_pred = model.predict(X_test)
            mlflow.log_param('n_estimators', n)
            mlflow.log_param('max_depth', d)
                    
            fbeta_score = cross_val_score(model, sm_X_train, sm_y_train, scoring = fbeta, cv = 5).mean()
            accuracy = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'accuracy', cv = 5).mean()
            precision = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'precision', cv = 5).mean() 
            recall = cross_val_score(model, sm_X_train, sm_y_train, scoring = 'recall', cv = 5).mean()
                    
            #accuracy = accuracy_score(y_test, y_pred)
            #precision = precision_score(y_test, y_pred)
            #recall = recall_score(y_test, y_pred)
            #fbeta = fbeta_score(y_test, y_pred, beta = 2)
                    
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("fbeta", fbeta_score)
                    
