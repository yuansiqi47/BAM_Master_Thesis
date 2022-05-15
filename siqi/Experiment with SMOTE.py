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

df = df.sample(frac=1,  random_state=42).reset_index(drop=True)

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

lr = LogisticRegression(random_state=0, penalty = 'none', max_iter = 2000)

# COMMAND ----------

def Smote_expriment(p):
    # smote
    sm = SMOTE(random_state=0, sampling_strategy = p)
    columns = X_train.columns
    sm_X_train, sm_y_train = sm.fit_resample(X_train, y_train)
    sm_X_train = pd.DataFrame(data=sm_X_train,columns=columns )
    sm_y_train = pd.DataFrame(data=sm_y_train,columns=['default'])
    sm_y_train = np.asarray(sm_y_train['default'])
    val = cross_val_score(lr, sm_X_train, sm_y_train, scoring = fbeta).mean()
    lr.fit(sm_X_train, sm_y_train)
    y_pred_lr = lr.predict(X_test)
    test = fbeta_score(y_test, y_pred_lr, beta = 2)
    return val, test

# COMMAND ----------

x = list(range(5, 105, 5))

# COMMAND ----------

val_score = []
test_score = []
for i in range(5, 105, 5):
    val, test = Smote_expriment(i/100)
    val_score.append(val)
    test_score.append(test)
    

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(x, val_score, label='validation')
plt.plot(x, test_score, label='test')
plt.title('Performance of logistic regression model')
plt.xlabel('proportion of defaults to non-defaults')
plt.ylabel('F-beta score')
plt.legend()
plt.show()

# COMMAND ----------

gb = GradientBoostingClassifier(random_state=0)

# COMMAND ----------

def Smote_gb_expriment(p):
    # smote
    sm = SMOTE(random_state=0, sampling_strategy = p)
    columns = X_train.columns
    sm_X_train, sm_y_train = sm.fit_resample(X_train, y_train)
    sm_X_train = pd.DataFrame(data=sm_X_train,columns=columns )
    sm_y_train = pd.DataFrame(data=sm_y_train,columns=['default'])
    sm_y_train = np.asarray(sm_y_train['default'])
    val = cross_val_score(gb, sm_X_train, sm_y_train, scoring = fbeta).mean()
    gb.fit(sm_X_train, sm_y_train)
    y_pred_gb = gb.predict(X_test)
    test = fbeta_score(y_test, y_pred_gb, beta = 2)
    return val, test

# COMMAND ----------

x = list(range(5, 105, 5))

# COMMAND ----------

val_score = []
test_score = []
for i in range(5, 105, 5):
    val, test = Smote_gb_expriment(i/100)
    val_score.append(val)
    test_score.append(test)
    

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(x, val_score, label='validation')
plt.plot(x, test_score, label='test')
plt.title('Performance of gradient boosting model')
plt.xlabel('proportion of defaults to non-defaults')
plt.ylabel('F-beta score')
plt.legend()
plt.show()

# COMMAND ----------


