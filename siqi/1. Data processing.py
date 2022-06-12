# Databricks notebook source
# Import packages
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats.mstats import winsorize
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table of content
# MAGIC #### 1. Predictors
# MAGIC 1.1 Overview of financial data 
# MAGIC 
# MAGIC 1.2 Financial ratios (predictors) calculation
# MAGIC 
# MAGIC 1.3 Data Cleaning (Winsorization)
# MAGIC #### 2. Merging predictors with defaults
# MAGIC 2.1 Making a subset of companies in the predictors dataset
# MAGIC 
# MAGIC 2.2 Selecting 'default' observations from 'delist'
# MAGIC 
# MAGIC 2.3 Linking company identifiers 'gvkey' and 'permno'
# MAGIC 
# MAGIC 2.4 Select 'default' observations from 'rating'
# MAGIC 
# MAGIC 2.5 Merging three data frames together
# MAGIC 
# MAGIC #### 3. Descriptive statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Predictors
# MAGIC ### 1.1 Overview of financial data

# COMMAND ----------

# load data
financials = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/compustat_financial_final.csv")
gdp =  pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/gdp_growth_rate_north_america.csv")

# COMMAND ----------

# take a look at the data 
financials.head()

# COMMAND ----------

# remove unnecessary columns 
financials = financials.drop(columns = ['indfmt','consol',	'popsrc',	'datafmt',	'tic','curcd','fyr','costat'])
# change the data type of the categorical variable
financials.gsector = financials.gsector.astype(object)

# COMMAND ----------

# take a look at the data summary
financials.groupby("fyear").mean().drop(columns = ['gvkey', 'datadate'])

# COMMAND ----------

# the size of the dataset
len(financials)

# COMMAND ----------

# the number of NA values for each column
financials.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Financial ratios (predictors) calculation

# COMMAND ----------

# exclude the outliers in 'emp' (i.e. exluding companies with 0 employee)
financials = financials[financials['emp'] > 0].reset_index(drop=True) 

# COMMAND ----------

# replace 0 in variables 'long-term debt' and 'short-term debt' with a small number 1
# to avoid some predictors being -Inf or Inf later
financials['dltt'] = financials['dltt'].replace(0, 1)

# COMMAND ----------

# creating variables for lag values
financials['roe'] = financials['ni']/financials['ceq']
# creating lag values
financials[['lag_rect', 'lag_at', 'lag_sale', 'lag_emp', 'lag_roe']] = financials.groupby("gvkey")[
    ['rect', 'at', 'sale', 'emp', 'roe']].shift(1)
# set the lag variable to NA if the year difference is not 1
financials[['lag_rect', 'lag_at', 'lag_sale', 'lag_emp', 'lag_roe']] = financials[
    ['lag_rect', 'lag_at', 'lag_sale', 'lag_emp', 'lag_roe']].where(financials.groupby('gvkey').fyear.diff() == 1,
                                                                              np.nan)

# COMMAND ----------

# Calculating ratios in liquidity 
financials['L1'] = financials['wcap']/financials['at']
financials['L2'] = financials['act']/(financials['at'] - financials['act'])
financials['L3'] = ((financials['ibc']+financials['dp'])/financials['at'])/financials['dltt']
financials['L4'] = (financials['oancf']+financials['ivncf']+financials['fincf'])/financials['sale']
financials['L5'] = financials['dlc']/financials['act']
financials['L6'] = financials['ch']/financials['at']
# Calculating ratios in profitability
financials['P1'] = financials['re']/financials['at']
financials['P2'] = financials['ni']/financials['at']
financials['P3'] = financials['ebitda']/financials['sale']
# Calculating ratios in efficiency 
financials['E1'] = financials['ebitda']/financials['at']
financials['E2'] = financials['sale']/financials['at']
financials['E3'] = ((financials['rect'] +financials['lag_rect'])/2)/financials['sale']
# Calculating ratios in leverage
financials['R1'] = (financials['pstkc'] +financials['csho'])/financials['at']
financials['R2'] = (financials['lt']-financials['che'])/financials['ebitda']
# Calculating ratios in growth
financials['G1'] = (financials['at']-financials['lag_at'])/financials['lag_at']
financials['G2'] = (financials['sale']-financials['lag_sale'])/financials['lag_sale']
financials['G3'] = financials['roe']-financials['lag_roe']
financials['G4'] = (financials['emp']-financials['lag_emp'])/financials['lag_emp']
# Calculating the ratio in size
# why we take log?
financials['S'] = np.log(financials['at'])

# COMMAND ----------

# getting the variables 
predictors = financials[['gvkey', 'datadate', 'fyear', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'P1', 'P2', 'P3', 
                         'E1', 'E2', 'E3', 'R1', 'R2', 'G1', 'G2', 'G3', 'G4', 'S', 'gsector']]

# COMMAND ----------

# merge the gdp dataset with the Compustat dataset
gdp.rename(columns = {'year': 'fyear'}, inplace =True)
predictors = pd.merge(predictors, gdp, on=['fyear'])

# COMMAND ----------

# have a look at the missing values
predictors.isnull().sum()

# COMMAND ----------

predictors = predictors.dropna(subset = ['gsector'])
len(predictors)

# COMMAND ----------

# test different thresholds for the number of NAs in each observation
print(len(predictors.dropna(thresh=21)))
print(len(predictors.dropna(thresh=22)))
print(len(predictors.dropna(thresh=23)))
print(len(predictors.dropna()))

# COMMAND ----------

# We choose the thresholds of 23 (i.e. maximum one missing value for each observation)
predictors = predictors.dropna(thresh=23).reset_index(drop=True)

# COMMAND ----------

# take a look at the number of missing values again
predictors.isnull().sum()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Merging predictors with defaults

# COMMAND ----------

# load data
delist = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/delist_V2.csv")
rating = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/rating_V2.csv")
link = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/link.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Making a subset of companies in the predictors dataset

# COMMAND ----------

len(predictors)

# COMMAND ----------

# select unique company identifier 'Permno' from data frame delist
permno_dic = np.unique(delist["PERMNO"].tolist())
# find the corresponding 'gvkey' based on the above unique 'permno'
gvkey_agg = link[link['LPERMNO'].isin(permno_dic)]['gvkey'].tolist()
# select unique company identifier 'gvkey' from data frame delist
for i in rating["gvkey"]:
    gvkey_agg.append(i)
# select 'gvkey's that at least apprear in one of the two data frames
gvkey_dic = np.unique(gvkey_agg)
# select a subset of predictors
predictors = predictors[predictors['gvkey'].isin(gvkey_dic)]
predictors = predictors.reset_index(drop=True)

# COMMAND ----------

len(predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Selecting 'default' observations from 'delist'

# COMMAND ----------

# select defaulted companies from the delist dataset and reset the index
delist_default = delist[(delist["DLSTCD"] < 500) | (delist["DLSTCD"].isin([560, 572, 574])) ]
delist_default = delist_default.reset_index(drop=True)

# change the data type
delist_default = delist_default.astype(int)
delist_default["DLSTDT"]= pd.to_datetime(delist_default["DLSTDT"], format='%Y%m%d')

# COMMAND ----------

# overview of the dataset
delist_default.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Linking company identifiers 'gvkey' and 'permno'

# COMMAND ----------

# change data type of 'link start date' and 'link end date'
link["LINKDT"]= pd.to_datetime(link["LINKDT"], format='%Y%m%d')
link["LINKENDDT"] = link["LINKENDDT"].replace('E', 20220316)
link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"], format='%Y%m%d')

# COMMAND ----------

# for each permno, if the maximum 'link end date' is earlier than today, that maximum 'link end date' is changed to today
# This is to aviod the case that PERMONO can't be linked with GVKEY because the 'link end date' is set too early 
for i in link['LPERMNO']:
    temp = link[link['LPERMNO'] == i]
    maxdate = max(temp["LINKENDDT"])
    link.loc[(link['LPERMNO'] == i) & (link["LINKENDDT"] == maxdate), "LINKENDDT"] = datetime.today()

# COMMAND ----------

# change datatype of 'link end date' to date
link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"]).dt.date

# COMMAND ----------

# link identifiers GVKEY and PERMNO and add GVKEY' to the data frame 'delist_default'
delist_default["gvkey"] = pd.Series(dtype= str)
for i in range(len(delist_default)):
    permno = delist_default.iloc[i]["PERMNO"]
    temp = link[link['LPERMNO'] == permno]
    if (len(temp) == 1):
        delist_default.loc[i,"gvkey"] = temp.iloc[0]["gvkey"]
    elif (len(temp) > 1):
        gvkey = None
        gvkey = temp[(temp["LINKDT"] < delist_default.iloc[i]["DLSTDT"]) & 
                     (temp["LINKENDDT"] > delist_default.iloc[i]["DLSTDT"])].iloc[0]["gvkey"]
        delist_default.loc[i,"gvkey"] = gvkey

# COMMAND ----------

# for PERMNO that can't be linked with GVKEY, observations are deleted
delist_default = delist_default[delist_default["gvkey"].notnull()]
delist_default = delist_default.reset_index(drop=True)

# COMMAND ----------

delist_default.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Select 'default' observations from 'rating'

# COMMAND ----------

# select defaulted companies from the rating dataset and reset the index
rating = rating[rating["splticrm"] == "D"]
rating = rating.drop_duplicates(subset = ['gvkey'], keep = 'first')
rating = rating.reset_index(drop=True)
rating

# COMMAND ----------

# change the data type of 'datadate' to date
rating["datadate"] = pd.to_datetime(rating["datadate"].astype('str'), format='%Y/%m/%d')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Merging three data frames together

# COMMAND ----------

# creating a dataframe df to store the merged data
df = predictors
# change the data type of 'datadate' to date
df["datadate"] = pd.to_datetime(df["datadate"], format='%Y%m%d')
# creating two variables for storing values of default
# e.g. if a company ends its fiscal year 2009 on 2010.04.30 and a 'default' was given in the period 2010.05.01 - 2011.04.30 in dataset 'rating', then we label the 'lead_rating' of corresponding observation in dataset predictors as 'default'.
df["lead_rating"] = pd.Series(dtype= str)
df["lead_delist"] = pd.Series(dtype= str)
# creating two variables for storing values of default dates
df['rating_date'] = pd.Series(dtype= 'datetime64[ns]')
df['delist_date'] = pd.Series(dtype= 'datetime64[ns]')

# COMMAND ----------

# merging 'df' together with 'rating'
for i in range(len(rating)):
    gvkey = rating.iloc[i]["gvkey"]
    date = rating.iloc[i]["datadate"]
    df.loc[(df["gvkey"] == gvkey) & (df["datadate"] < date ) &
            (df["datadate"] + pd.offsets.DateOffset(years=1) > date), "lead_rating"] = "D"
    df.loc[(df["gvkey"] == gvkey) & (df["datadate"] < date ) &
            (df["datadate"] + pd.offsets.DateOffset(years=1) > date), "rating_date"] = date

# COMMAND ----------

# merging 'df' together with 'delist_default'
for i in range(len(delist_default)):
    gvkey = delist_default.iloc[i]["gvkey"]
    date = delist_default.iloc[i]["DLSTDT"]
    df.loc[(df["gvkey"] == gvkey) & (df["datadate"] < date ) &
            (df["datadate"] + pd.offsets.DateOffset(years=1) > date), "lead_delist"] = "D"
    df.loc[(df["gvkey"] == gvkey) & (df["datadate"] < date ) &
            (df["datadate"] + pd.offsets.DateOffset(years=1) > date), "delist_date"] = date

# COMMAND ----------

# create a new variable 'default'
def label_default (row):
   if row['lead_rating'] == "D" :
      return 1
   if row['lead_delist'] == "D" :
      return 1
   return 0
df['default'] = df.apply (lambda row: label_default(row), axis=1)

# COMMAND ----------

# getting the earlier date of default
df['default_date'] = df[["rating_date","delist_date"]].min(axis=1)
# Because some companies could be labeled as defaults for years, 
# only the observation firstly labelled as 'default' is kept.
default = df[df["default"] == 1].drop_duplicates(subset = ['gvkey'], keep = 'first')

# COMMAND ----------

# take a look at the default observations
default[['gvkey','datadate','fyear','lead_rating','lead_delist','default']].head(10)

# COMMAND ----------

# recombine non-default and default observations together
df = pd.concat([df[df["default"] == 0], default]).sort_values(by = ['gvkey','datadate']).reset_index(drop=True)
# drop columns 'lead_rating' and 'lead_delist'
df = df.drop(columns = ['lead_rating','lead_delist', 'rating_date', 'delist_date'])

# COMMAND ----------

df.head()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Final data processing and descriptive statistics

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 industry summary

# COMMAND ----------

# remove the observations from financials, utilities and real-estate industries, because those are highly regulated
df = df.drop(index = df.index[df['gsector'].isin([40, 55, 60])]).reset_index(drop=True) 

# COMMAND ----------

# get a data summary of the industry
defaults_num = df[df['default'] ==1].groupby(['gsector'])['gsector'].count().tolist()
observations_num = df.groupby(['gsector'])['gsector'].count().tolist()
companies_num = df.drop_duplicates(subset = ['gvkey'], keep = 'first').groupby(['gsector'])['gsector'].count().tolist()
industry = ["Engergy", "Materials", "Industries", "Consumer Discretionary", "Consumer Staples", "Health Care", 
            "Information Technology", "Communication Services"]
pd.DataFrame({'defaults': defaults_num, 'companies': companies_num, 'observations': observations_num}, index = industry)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 winsorize all variables to get rid of extreme values

# COMMAND ----------

df.describe()

# COMMAND ----------

def using_mstats(s):
    '''define a function to winsorize series in dataframe (90% winsorization)
        (sets all observations greater than the 95th percentile equal to the value at the 95th percentile 
        and all observations less than the 5th percentile equal to the value at the 5th percentile.)'''
    s[s.notna()] = winsorize(s[s.notna()], limits=[0.05, 0.05])
    return s


# COMMAND ----------

# winsorize all predictors
df.iloc[:,3:22] = df.iloc[:,3:22].apply(using_mstats, axis=0)
df.iloc[:,3:22].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Multicollinearity check 

# COMMAND ----------

fig, heat_map = plt.subplots(figsize=(20,20))
corr = df.drop(columns = ['gvkey', 'fyear', 'default', 'gsector']).corr()
heat_map = sns.heatmap(corr, linewidth = 1 , annot = True)

# COMMAND ----------

# from the graph, we drop P2 
df = df.drop(columns = ['P2'])

# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 default composition 

# COMMAND ----------

print("The number of all observations:", len(df))
print("The number of unique companies: ", len(df['gvkey'].unique()))
print("the number of default observations: ", len(df[df['default'] == 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 statistical summary

# COMMAND ----------

summary = df.describe().drop(columns = ['gvkey', 'fyear', 'default']).round(3).T
summary['count'] = round((len(df) - summary['count'])/len(df)*100, 2).astype(str) + '%'
summary = summary.rename(columns = {'count': 'NA'})
summary

# COMMAND ----------

print(summary.to_latex())

# COMMAND ----------

# write csv file
df.to_csv('/dbfs/FileStore/Siqi thesis/df.csv', index=False)

# COMMAND ----------



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

df = df.sample(frac=1,  random_state=42).reset_index(drop=True)

# COMMAND ----------

X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date'], axis = 1)
y = df['default']

# COMMAND ----------

X = pd.get_dummies(X, columns = ['gsector'])

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

fbeta = make_scorer(fbeta_score, beta=2)
metrics = {'fbeta': fbeta, 'accuracy':'accuracy', 'precision':'precision', 'recall':'recall'}

# COMMAND ----------

log_model = statm.Logit(sm_y_train, sm_X_train)

# COMMAND ----------

log_model.fit().summary()

# COMMAND ----------

lr = LogisticRegression(random_state=0, penalty = 'none', max_iter = 1000)
lr.fit(sm_X_train, sm_y_train)
y_pred_lr = lr.predict(X_test)
model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
evaluation = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
                         index = model)
evaluation.loc['Logistic regression (baseline)', :] = [accuracy_score(y_test, y_pred_lr),
                                                       precision_score(y_test, y_pred_lr), 
                                                       recall_score(y_test, y_pred_lr), 
                                                       fbeta_score(y_test, y_pred_lr, beta = 2)]
evaluation

# COMMAND ----------

lr = RandomForestClassifier(max_depth=20.0, n_estimators=591, random_state=0)
lr.fit(sm_X_train, sm_y_train)
y_pred_lr = lr.predict(X_test)
#model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
#evaluation = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'F-beta'],
#                         index = model)
evaluation.loc['Random Forest', :] = [accuracy_score(y_test, y_pred_lr),
                                                       precision_score(y_test, y_pred_lr), 
                                                       recall_score(y_test, y_pred_lr), 
                                                       fbeta_score(y_test, y_pred_lr, beta = 2)]
evaluation

# COMMAND ----------


