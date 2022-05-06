# Databricks notebook source
# MAGIC %md
# MAGIC ## Testing how many defaults could be linked with observations in Dealscan

# COMMAND ----------

pip install openpyxl

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# load data
df_pd = pd.read_csv("/dbfs/FileStore/Siqi thesis/df_pd.csv")

# COMMAND ----------

df_pd  = df_pd .iloc[:,1:]

# COMMAND ----------

df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------merging-----------------------------------

# COMMAND ----------

dealscan_link = pd.read_excel("/dbfs/FileStore/Siqi thesis/raw data/dealscan_link.xlsx")
lender = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/lender.csv")
loan = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/loan.csv")

# COMMAND ----------

loan

# COMMAND ----------

dealscan_link = dealscan_link[['facid', 'bcoid', 'gvkey']]
loan = loan[['FacilityID', 'BorrowerCompanyID', 'FacilityStartDate', 'FacilityAmt', 'Currency', 'ExchangeRate', 'Maturity']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_link, loan, how="inner", left_on= ['facid', 'bcoid'] ,right_on=['FacilityID','BorrowerCompanyID'])

# COMMAND ----------

dealscan_merged["FacilityStartDate"] = pd.to_datetime(dealscan_merged["FacilityStartDate"], format='%Y%m%d')
df_pd["datadate"] = pd.to_datetime(df_pd["datadate"], format='%Y-%m-%d')

# COMMAND ----------

dealscan_merged

# COMMAND ----------

dealscan_merged[['fyear', 'datadate', 'default', 'default_date', 'pred_lr', 'pd_lr']] = ''

# COMMAND ----------

for i in range(len(dealscan_merged)):
    gvkey = dealscan_merged.iloc[i]["gvkey"]
    date = dealscan_merged.iloc[i]["FacilityStartDate"]
    temp_df = df_pd[(df_pd["gvkey"] == gvkey) & 
                    (df_pd["datadate"] < date ) &
                    (df_pd["datadate"] + pd.offsets.DateOffset(years=1) > date)]
    if len(temp_df) == 0:
        continue
    else:
        dealscan_merged.iloc[i, -6:] = temp_df.iloc[0, 1:]

        
#dealscan_merged.iloc[i, -4] = df[(df["gvkey"] == gvkey) & (df["datadate"] < date ) & (df["datadate"] + pd.offsets.DateOffset(years=1) > date)].iloc[:, 1:]

# COMMAND ----------

dealscan_merged

# COMMAND ----------

dealscan_merged= dealscan_merged.drop(index = dealscan_merged.index[dealscan_merged['default'] == ''])
dealscan_merged

# COMMAND ----------

dealscan_merged = dealscan_merged.dropna(subset = ['Maturity'])

# COMMAND ----------

len(dealscan_merged)

# COMMAND ----------

dealscan_merged['default'].sum()

# COMMAND ----------

len(dealscan_merged[dealscan_merged['default'] == 1])

# COMMAND ----------

len(dealscan_merged[(dealscan_merged['default'] == 1) & (dealscan_merged['default_date'] > dealscan_merged['FacilityStartDate'] )])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### CR calculation

# COMMAND ----------

import numpy as np
import math
from scipy.stats import norm

# COMMAND ----------

def Risk_weight(PD, M):
    LGD = 1
    R = 0.12*(1-math.exp(-50*PD))/(1-math.exp(-50))+0.24*(1-math.exp(-50*PD))/(1-math.exp(-50))
    b = (0.11852-0.05478*np.log(PD))**2
    RW = (LGD*norm.cdf(1/np.sqrt(1-R)*norm.ppf(PD)+1/np.sqrt(1-R)*norm.ppf(0.999))-LGD*PD)*(1+(M-2.5*b))/(1-1.5*b)*1.25*1.06
    return RW

# COMMAND ----------

rw_lr = list(map(Risk_weight, dealscan_merged['pd_lr'].tolist(), dealscan_merged['Maturity'].tolist()))
rw_lr

# COMMAND ----------

dealscan_merged['rw_lr'] = rw_lr

# COMMAND ----------

dealscan_merged['cr_lr'] = 0.08 * dealscan_merged['rw_lr'] * dealscan_merged['FacilityAmt'] * dealscan_merged['ExchangeRate']

# COMMAND ----------

dealscan_merged['cr_lr'].sum()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Economic effects (gain/loss)

# COMMAND ----------

interest = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/interest.csv")

# COMMAND ----------

interest

# COMMAND ----------

interest = interest.dropna(subset = ['BaseRate'])
interest['rate'] = (interest['MinBps'] + interest['MaxBps'])/2/10000
interest = interest.groupby('FacilityID').mean()[['rate']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_merged, interest, how="left",on='FacilityID')
dealscan_merged

# COMMAND ----------

len(dealscan_merged)

# COMMAND ----------

len(dealscan_merged.dropna(subset = ['rate']))

# COMMAND ----------

dealscan_merged = dealscan_merged.dropna(subset = ['rate'])

# COMMAND ----------

def Eco_loss(df, pred_model):
    df_fn = df[(df['default'] == 1) & (df[pred_model] == 0)]
    loss_fn = sum(df_fn['FacilityAmt'] * df_fn['ExchangeRate'])
    df_fp = df[(df['default'] == 0) & (df[pred_model] == 1)]
    loss_fp = sum(df_fp['FacilityAmt'] * df_fp['ExchangeRate'] *df_fp['rate'])
    loss_cr = 0.115 * df['cr_lr'].sum()
    loss = loss_fn + loss_fp + loss_cr
    return loss

# COMMAND ----------

Eco_loss(dealscan_merged, 'pred_lr')

# COMMAND ----------

(#-------------------------------break line--------------------------------

# COMMAND ----------

count

# COMMAND ----------

lender

# COMMAND ----------

loan

# COMMAND ----------

df["matched"] = pd.Series(dtype= int)
for i in range(len(trydf)):
    gvkey = trydf.iloc[i]["gvkey"]
    date = trydf.iloc[i]["FacilityStartDate"]
    df.loc[(df["gvkey"] == gvkey) & (df["datadate"] - pd.offsets.DateOffset(years=1) < date ) &
            (df["datadate"]  > date), "matched"] = 1

# COMMAND ----------

df[(df["matched"] == 1) & (df["default"] == 1)].head(50)

# COMMAND ----------

df["datadate"] = pd.to_datetime(df["datadate"], format='%Y-%m-%d')

# COMMAND ----------

trydf['FacilityStartDate']

# COMMAND ----------


