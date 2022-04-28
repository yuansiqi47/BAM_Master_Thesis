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
df = pd.read_csv("/dbfs/FileStore/Thesis data/df_V2.csv")

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
# MAGIC ### Lasso penalized logistic regression

# COMMAND ----------

param_grid_lasso = {'C' : [0.001, 0.01, 0.1, 0.2, 0.5, 1]}
                   #'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

# COMMAND ----------

lasso = LogisticRegression(random_state=0, penalty = 'l1', solver = 'liblinear', max_iter=1000)
grid_search_lasso = GridSearchCV(estimator = lasso, param_grid = param_grid_lasso, scoring = fbeta, cv = 10)

# COMMAND ----------

grid_search_lasso.fit(sm_X_train, sm_y_train)

# COMMAND ----------

best_lasso = grid_search_lasso.best_estimator_
best_lasso.fit(sm_X_train, sm_y_train)

# COMMAND ----------

grid_search_lasso.best_score_

# COMMAND ----------

y_pred = best_lasso.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)

# COMMAND ----------

evaluation.loc['Logistic regression (Lasso)', :] = [accuracy_score(y_test, y_pred),
                                                       precision_score(y_test, y_pred), 
                                                       recall_score(y_test, y_pred), 
                                                       fbeta_score(y_test, y_pred, beta = 2)]
evaluation

# COMMAND ----------

grid_search_lasso.cv_results_

# COMMAND ----------

##############################################stop here#####################################################

# COMMAND ----------

import mlflow

# COMMAND ----------

with mlflow.start_run(run_name='lasso') as run:
 # model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)

  param_grid_lasso = {'C' : [0.2, 0.5, 1]}
    
  lasso = LogisticRegression(random_state=0, penalty = 'l1', solver = 'saga')

  grid_search_lasso = GridSearchCV(estimator = lasso, param_grid = param_grid_lasso, scoring = fbeta, cv = 10)
    
  # Models, parameters, and training metrics are tracked automatically
  grid_search_lasso.fit(sm_X_train, sm_y_train)
    
  best_lasso = grid_search_lasso.best_estimator_
  best_lasso.fit(sm_X_train, sm_y_train)
  y_pred = best_lasso.predict(X_test)
#  predicted_probs = best_lasso.predict_proba(X_test)
#  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  fbeta_score = fbeta_score(y_test, y_pred, beta = 2)
    
  # The AUC score on test data is not automatically logged, so log it manually
  # mlflow.log_metric("test_auc", roc_auc)
  # print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Support Vector Machine

# COMMAND ----------

param_grid_svm = {'C' : [0.01, 0.1, 0.2, 0.5, 1],
                  'kernel': ['linear','poly', 'rbf'],
                  'degree': [2, 3, 4, 5]
                 }

# COMMAND ----------

svm = SVC(random_state = 0)
grid_search_svm = GridSearchCV(estimator = svm, param_grid = param_grid_svm, scoring = fbeta, cv = 10)

# COMMAND ----------

# sm_X_train2, sm_X_test2, sm_y_train2, sm_y_test2 = train_test_split(sm_X_train, sm_y_train, stratify = sm_y_train, test_size = 0.9, random_state=0)

# COMMAND ----------

grid_search_svm.fit(sm_X_train, sm_y_train)

# COMMAND ----------

grid_search_svm.best_score_

# COMMAND ----------

best_svm = grid_search_svm.best_estimator_
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

grid_search_svm.cv_results_

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Random Forest

# COMMAND ----------

param_grid_rf = {'n_estimators' : [50, 100, 200, 250, 300],
                  'max_depth': [4, 6, 8, 15],
                  'min_samples_split': [20, 50, 100],
                  'min_samples_leaf': [10, 20, 50]
                 }

# COMMAND ----------

rf = RandomForestClassifier(random_state=0)
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, scoring = fbeta, cv = 10)

# COMMAND ----------

grid_search_rf.fit(sm_X_train, sm_y_train)

# COMMAND ----------

grid_search_rf.best_score_

# COMMAND ----------

best_rf = grid_search_rf.best_estimator_
best_rf.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred = best_rf.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)

# COMMAND ----------

evaluation.loc['Random Forest', :] = [accuracy_score(y_test, y_pred),
                                      precision_score(y_test, y_pred), 
                                      recall_score(y_test, y_pred),
                                      fbeta_score(y_test, y_pred, beta = 2)]
evaluation

# COMMAND ----------

grid_search_rf.cv_results_

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Gradient boosting

# COMMAND ----------

param_grid_gb = {'n_estimators' : [50, 100, 200, 300],
                  'max_depth': [1, 2, 4, 6, 10],
                  'learning_rate': [0.001, 0.01, 0.1, 0.5]
                 }

# COMMAND ----------

gb = GradientBoostingClassifier(random_state=0)
grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid_gb, scoring = fbeta, cv = 10)

# COMMAND ----------

grid_search_gb.fit(sm_X_train, sm_y_train)

# COMMAND ----------

grid_search_gb.best_score_

# COMMAND ----------

best_gb = grid_search_gb.best_estimator_
best_gb.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred = best_gb.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)

# COMMAND ----------

evaluation.loc['Gradient Boosting', :] = [accuracy_score(y_test, y_pred),
                                          precision_score(y_test, y_pred), 
                                          recall_score(y_test, y_pred),
                                          fbeta_score(y_test, y_pred, beta = 2)]
evaluation

# COMMAND ----------

grid_search_gb.cv_results_

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Legacy

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

# Enable MLflow autologging for this notebook
mlflow.autolog()

# COMMAND ----------

# load data
df = pd.read_csv("/dbfs/FileStore/Thesis data/df_V2.csv")
df = df.drop(columns = ['P1','P2', 'E1'])
df = df.iloc[:,1:]
df = df.dropna(axis=0, how='any')
X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date', 'gsector'], axis = 1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = df['default'], test_size = 0.3, random_state=0)

#X_train = X_train.sample(frac=1,  random_state=42).reset_index(drop=True)
#y_train = y_train.sample(frac=1,  random_state=42).reset_index(drop=True)

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
 # model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)

  param_grid_gb = {'n_estimators' : [50, 100, 200, 300],
                  'max_depth': [1, 2, 4, 6, 10],
                  'learning_rate': [ 0.01, 0.1, 0.5]
                 }
  gb = sklearn.ensemble.GradientBoostingClassifier(random_state=0)

  grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid_gb, scoring = scoring_metrics, refit = False,
                           cv = 10, n_jobs = -1, return_train_score =True)
  # Models, parameters, and training metrics are tracked automatically
  grid_search_gb.fit(sm_X_train, sm_y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # The AUC score on test data is not automatically logged, so log it manually
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

from time import time

# COMMAND ----------

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
 #   results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
#    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
#    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
 #   results['f_train'] = fbeta_score(y_train[:300], predictions_train[:300], beta =2)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 2)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

# COMMAND ----------

# Initialize the three models
clf_1 = SVC(random_state=0)
clf_2 = RandomForestClassifier(random_state=0)
clf_3 = GradientBoostingClassifier(random_state=0)
clf_4 = LogisticRegression(random_state=0)

# Calculate the number of samples for 0.01%, 0.1%, 1%, 10%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = int(0.1*len(y_train))
samples_1 = int(0.01*len(y_train))
samples_01 = int(0.001*len(y_train))
samples_001 = int(0.0001*len(y_train))

# Collect results on the learners
results = {}
for clf in [clf_1, clf_2, clf_3, clf_4]:
    clf_name = clf.__class__.__name__
    results[clf_name] = train_predict(clf, samples_100, sm_X_train, sm_y_train, X_test, y_test)
#    results[clf_name] = {}
 #   for i, samples in enumerate([samples_001, samples_01, samples_1]):
  #      results[clf_name][i] = \
   #     train_predict(clf, samples, sm_X_train, sm_y_train, X_test, y_test)



# COMMAND ----------

results

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Lasso penalized logistic regression

# COMMAND ----------

param_grid_lasso = {'C' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]}

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'l2', l1_ratio=0)
grid_search = GridSearchCV(estimator = lr, param_grid = param_grid_lasso, scoring = scoring_metrics, refit = False,
                           cv = 10, return_train_score =True)

# COMMAND ----------

grid_search.fit(sm_X_train, sm_y_train)

# COMMAND ----------

param_range = np.array(list(param_grid_lasso.values())[0])

# COMMAND ----------

def validation_curve(name, hyperparameter, param_range):
    accuracy = grid_search.cv_results_['mean_test_accuracy']
    accuracy_std = grid_search.cv_results_['std_test_accuracy']
    precision = grid_search.cv_results_['mean_test_precision']
    precision_std = grid_search.cv_results_['std_test_precision']
    recall = grid_search.cv_results_['mean_test_recall']
    recall_std = grid_search.cv_results_['std_test_recall']
    f1 = grid_search.cv_results_['mean_test_f1']
    f1_std = grid_search.cv_results_['std_test_f1']
    roc_auc = grid_search.cv_results_['mean_test_roc_auc']
    roc_auc_std = grid_search.cv_results_['std_test_roc_auc']

    plt.title("Validation Curve" + name)
    plt.xlabel(hyperparameter)
    plt.ylabel("Score")
    plt.ylim(0.7, 0.9)
    lw = 2
    plt.semilogx(
        param_range, accuracy, label="accuracy", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        accuracy - accuracy_std ,
        accuracy + accuracy_std ,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, precision, label="precision", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        precision - precision_std,
        precision + precision_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.semilogx(
        param_range, recall, label="recall", color="darkviolet", lw=lw
    )
    plt.fill_between(
        param_range,
        recall - recall_std,
        recall + recall_std,
        alpha=0.2,
        color="darkviolet",
        lw=lw,
    )
    plt.semilogx(
        param_range, f1, label="f1", color="olive", lw=lw
    )
    plt.fill_between(
        param_range,
        f1 - f1_std,
        f1 + f1_std,
        alpha=0.2,
        color="olive",
        lw=lw,
    )
    plt.semilogx(
        param_range, roc_auc, label="roc_auc", color="darkred", lw=lw
    )
    plt.fill_between(
        param_range,
        roc_auc - roc_auc_std,
        roc_auc + roc_auc_std,
        alpha=0.2,
        color="darkred",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()


# COMMAND ----------

validation_curve('Lasso', 'Penalty', param_range)

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'l2', l1_ratio=0, C = 0.001)
lr.fit(sm_X_train, sm_y_train)

# COMMAND ----------

y_pred = lr.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, y_pred)
