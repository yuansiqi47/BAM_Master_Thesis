# Databricks notebook source
# MAGIC %md
# MAGIC # Assessing and Optimizing Default Prediction Models
# MAGIC Siqi Yuan \
# MAGIC Credit Risk | ART: AI in Financial Service

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Context
# MAGIC ### 1. What
# MAGIC * Banks build their own prediction models to better screen customer defaults.
# MAGIC * In the past decades, traditional statistical models like logistic regression are heavily implemented.
# MAGIC * Nowadays, practioners have thrown light on ML methods.
# MAGIC 
# MAGIC Research Question: **What ML models perform better than traditional models in default prediction?**
# MAGIC 
# MAGIC ### 2. Research objects
# MAGIC * Corporate borrowers (Retail borrowers are out of scope)
# MAGIC * Financial data and default record of listed companies in US from 2010 to 2021 
# MAGIC 
# MAGIC ### 3. Goal
# MAGIC To showcase how to make use of ML and modernize credit risk modelling in banks
# MAGIC 
# MAGIC ### 4. Method
# MAGIC * Overall approach: 
# MAGIC     - Baseline model: simple logistic regression model
# MAGIC     - ML models: lasso penalized logistic regression, support vector machine, random forest, and gradient boosting
# MAGIC * Roadmap
# MAGIC    - <img src="files/visuals/roadmap.png" width=800/></a> 
# MAGIC    
# MAGIC ### 5. Data
# MAGIC * <mark>financials</mark>: yearly accounting data of listed companies in US from 2010 to 2021 
# MAGIC * <mark>delist</mark>: records of companies delisted in US from 2010 to 2021 because of default
# MAGIC * <mark>rating</mark>: records of companies labeled as "default" by a thrid-party authority 
# MAGIC 
# MAGIC ### 6. Data merging steps
# MAGIC * Data overview
# MAGIC * Make a subset of 'financials'
# MAGIC <img src="files/visuals/fin_subset.png" width=400/></a> 
# MAGIC * Select observations labeled as 'default' of 'delist'
# MAGIC * Select observations labeled as 'default' of 'rating'
# MAGIC * Merging three data frames together
# MAGIC 
# MAGIC ### 7. Outcome
# MAGIC * a final data frame with all variables in 'financials' and an indicator of 'default' of the upcoming year

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Merging

# COMMAND ----------

# Import packages
import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# load data
financials = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/compustat_financial_copy.csv")
delist = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/delist.csv")
rating = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/rating.csv")
link = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/link.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Data overview

# COMMAND ----------

financials[['gvkey','datadate','fyear']].head()

# COMMAND ----------

delist.head()

# COMMAND ----------

delist[(delist["DLSTCD"] < 500) | (delist["DLSTCD"].isin([560, 572, 574])) ].head()

# COMMAND ----------

link[['gvkey','LPERMNO','LINKDT','LINKENDDT']].head()

# COMMAND ----------

rating[rating["splticrm"] == "D"][['gvkey','splticrm','datadate']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Make a subset of 'financials'
# MAGIC <img src="files/visuals/fin_subset.png" width=400/></a> 

# COMMAND ----------

len(financials)

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
# select a subset of financials
financials = financials[financials['gvkey'].isin(gvkey_dic)]
financials = financials.reset_index(drop=True)

# COMMAND ----------

len(financials)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Select 'default' observations of 'delist'

# COMMAND ----------

delist_default = delist[(delist["DLSTCD"] < 500) | (delist["DLSTCD"].isin([560, 572, 574])) ]
delist_default = delist_default.reset_index(drop=True)
delist_default = delist_default.astype(int)
delist_default["DLSTDT"]= pd.to_datetime(delist_default["DLSTDT"], format='%Y%m%d')

# COMMAND ----------

delist_default

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. link identifiers 'gvkey' and 'permno' and add 'gvkey' to the data frame 'delist_default'

# COMMAND ----------

# cleaning data of link start date and link end date
link["LINKDT"]= pd.to_datetime(link["LINKDT"], format='%Y%m%d')
link["LINKENDDT"] = link["LINKENDDT"].replace('E', 20220316)
link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"], format='%Y%m%d')

# COMMAND ----------

# cleaning data of link start date and link end date
for i in link['LPERMNO']:
    temp = link[link['LPERMNO'] == i]
    maxdate = max(temp["LINKENDDT"])
    link.loc[(link['LPERMNO'] == i) & (link["LINKENDDT"] == maxdate), "LINKENDDT"] = datetime.today()

# COMMAND ----------

# change datatype of link end date to date
link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"]).dt.date

# COMMAND ----------

# link identifiers 'gvkey' and 'permno' and add 'gvkey' to the data frame 'delist_default'
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

delist_default.head()

# COMMAND ----------

delist_default = delist_default[delist_default["gvkey"].notnull()]
delist_default = delist_default.reset_index(drop=True)

# COMMAND ----------

delist_default.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Select 'default' observations of 'rating'

# COMMAND ----------

rating = rating[rating["splticrm"] == "D"]
rating = rating.reset_index(drop=True)
rating

# COMMAND ----------

rating["datadate"] = pd.to_datetime(rating["datadate"].astype('str'), format='%Y/%m/%d')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Merging three data frames together

# COMMAND ----------

financials["datadate"] = pd.to_datetime(financials["datadate"], format='%Y%m%d')
financials["lead_rating"] = pd.Series(dtype= str)
financials["lead_delist"] = pd.Series(dtype= str)
# e.g. if a company ends its fiscal year 2009 on 2010.04.30 and a 'default' was given in the period 2010.05.01 - 2011.04.30 in dataset 'rating', then we label the 'lead_rating' of corresponding observation in dataset financials as 'default'.

# COMMAND ----------

# merging 'financials' together with 'rating'
for i in range(len(rating)):
    gvkey = rating.iloc[i]["gvkey"]
    date = rating.iloc[i]["datadate"]
    financials.loc[(financials["gvkey"] == gvkey) & (financials["datadate"] < date ) &
            (financials["datadate"] + pd.offsets.DateOffset(years=1) > date), "lead_rating"] = "D"

# COMMAND ----------

# merging 'financials' together with 'delist_default'
for i in range(len(delist_default)):
    gvkey = delist_default.iloc[i]["gvkey"]
    date = delist_default.iloc[i]["DLSTDT"]
    financials.loc[(financials["gvkey"] == gvkey) & (financials["datadate"] < date ) &
            (financials["datadate"] + pd.offsets.DateOffset(years=1) > date), "lead_delist"] = "D"

# COMMAND ----------

financials[financials["lead_delist"] == "D"].head()

# COMMAND ----------

financials[financials["lead_rating"] == "D"].head()

# COMMAND ----------

# create a new variable 'default'
def label_default (row):
   if row['lead_rating'] == "D" :
      return 1
   if row['lead_delist'] == "D" :
      return 1
   return 0
financials['default'] = financials.apply (lambda row: label_default(row), axis=1)

# COMMAND ----------

default = financials[financials["default"] == 1].drop_duplicates(subset = ['gvkey'], keep = 'first')

# COMMAND ----------

default[['gvkey','datadate','fyear','lead_rating','lead_delist','default']]

# COMMAND ----------

default.isnull().sum()

# COMMAND ----------

default.drop(columns = ['opiti', 'ugi','xlr', 'mkvalt','lead_rating','lead_delist']).dropna()

# COMMAND ----------

link.to_csv('/dbfs/FileStore/Thesis data/example.csv')

# COMMAND ----------


