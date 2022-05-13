# Databricks notebook source
# MAGIC %md
# MAGIC ## Economical analysis

# COMMAND ----------

pip install openpyxl

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# load data
df_pd = pd.read_csv("/dbfs/FileStore/Siqi thesis/df_pd.csv")

# COMMAND ----------

#df_pd  = df_pd .iloc[:,3:]

# COMMAND ----------

df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------merging-----------------------------------

# COMMAND ----------

dealscan_link = pd.read_excel("/dbfs/FileStore/Siqi thesis/raw data/dealscan_link.xlsx")
loan = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/loan.csv")
interest = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/interest.csv")

# COMMAND ----------

loan.head()

# COMMAND ----------

dealscan_link = dealscan_link[['facid', 'bcoid', 'gvkey']]
loan = loan[['FacilityID', 'BorrowerCompanyID', 'FacilityStartDate', 'FacilityAmt', 'Currency', 'ExchangeRate', 'Maturity']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_link, loan, how="inner", left_on= ['facid', 'bcoid'] ,right_on=['FacilityID','BorrowerCompanyID'])

# COMMAND ----------

dealscan_merged["FacilityStartDate"] = pd.to_datetime(dealscan_merged["FacilityStartDate"], format='%Y%m%d')
df_pd["datadate"] = pd.to_datetime(df_pd["datadate"], format='%Y-%m-%d')

# COMMAND ----------

dealscan_merged = dealscan_merged.drop(columns = ['facid','bcoid'])

# COMMAND ----------

dealscan_merged[['fyear', 'datadate', 'default', 'default_date', 'pred_lr', 'pd_lr', 
                 'pred_lasso', 'pd_lasso', 'pred_svm', 'pd_svm', 'pred_rf', 'pd_rf', 
                 'pred_gb', 'pd_gb']] = ''

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
        dealscan_merged.iloc[i, -14:] = temp_df.iloc[0, 1:]

        
#dealscan_merged.iloc[i, -4] = df[(df["gvkey"] == gvkey) & (df["datadate"] < date ) & (df["datadate"] + pd.offsets.DateOffset(years=1) > date)].iloc[:, 1:]

# COMMAND ----------

dealscan_merged.head()

# COMMAND ----------

dealscan_merged = dealscan_merged.drop(index = dealscan_merged.index[dealscan_merged['default'] == ''])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Merge with interest

# COMMAND ----------

interest.head()

# COMMAND ----------

interest = interest.dropna(subset = ['BaseRate'])
interest['rate'] = (interest['MinBps'] + interest['MaxBps'])/2/10000
interest = interest.groupby('FacilityID').mean()[['rate']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_merged, interest, how="left",on='FacilityID')
dealscan_merged.head()

# COMMAND ----------

dealscan_merged = dealscan_merged.dropna(subset = ['rate', 'Maturity']).reset_index(drop = True)

# COMMAND ----------

len(dealscan_merged)

# COMMAND ----------

dealscan_merged['default'].sum()

# COMMAND ----------

len(dealscan_merged[(dealscan_merged['default'] == 1) & (dealscan_merged['default_date'] > dealscan_merged['FacilityStartDate'] )])

# COMMAND ----------

dealscan_merged.to_csv('/dbfs/FileStore/Siqi thesis/dealscan_merged.csv', index=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### CR calculation

# COMMAND ----------

import numpy as np
import math
from scipy.stats import norm

# COMMAND ----------

def Captial_Requirement(PD, M, loan_amt, exchange_rate):
    if PD == 0:
        return 0
    else:
        LGD = 1
        R = 0.12*(1-math.exp(-50*PD))/(1-math.exp(-50))+0.24* (1 - (1-math.exp(-50*PD))/(1-math.exp(-50)))
        b = (0.11852-0.05478*np.log(PD))**2
        RW = (LGD*norm.cdf(1/np.sqrt(1-R)*norm.ppf(PD)+np.sqrt(R/(1-R))*norm.ppf(0.999))-LGD*PD)*(1+(M-2.5)*b)/(1-1.5*b)*1.25*1.06
        CR = RW * loan_amt * exchange_rate * 0.08
        return CR

# COMMAND ----------

CR_lr = list(map(Captial_Requirement, dealscan_merged['pd_lr'].tolist(), dealscan_merged['Maturity'].tolist(),
                dealscan_merged['FacilityAmt'].tolist(), dealscan_merged['ExchangeRate'].tolist()))
CR_lasso = list(map(Captial_Requirement, dealscan_merged['pd_lasso'].tolist(), dealscan_merged['Maturity'].tolist(),
                dealscan_merged['FacilityAmt'].tolist(), dealscan_merged['ExchangeRate'].tolist()))
CR_svm = list(map(Captial_Requirement, dealscan_merged['pd_svm'].tolist(), dealscan_merged['Maturity'].tolist(),
                dealscan_merged['FacilityAmt'].tolist(), dealscan_merged['ExchangeRate'].tolist()))
CR_rf = list(map(Captial_Requirement, dealscan_merged['pd_rf'].tolist(), dealscan_merged['Maturity'].tolist(),
                dealscan_merged['FacilityAmt'].tolist(), dealscan_merged['ExchangeRate'].tolist()))
CR_gb = list(map(Captial_Requirement, dealscan_merged['pd_gb'].tolist(), dealscan_merged['Maturity'].tolist(),
                dealscan_merged['FacilityAmt'].tolist(), dealscan_merged['ExchangeRate'].tolist()))

# COMMAND ----------

dealscan_merged['CR_lr'] = CR_lr
dealscan_merged['CR_lasso'] = CR_lasso
dealscan_merged['CR_svm'] = CR_svm
dealscan_merged['CR_rf'] = CR_rf
dealscan_merged['CR_gb'] = CR_gb

# COMMAND ----------

model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
Eco_evaluation = pd.DataFrame(columns = model,
                         index = ['Capital requirement (billion)', 'Economical loss (billion)'])

# COMMAND ----------

Eco_evaluation.loc['Capital requirement (billion)', :] = [round(dealscan_merged['CR_lr'].sum()/1000000000, 3),
                                                          round(dealscan_merged['CR_lasso'].sum()/1000000000, 3),
                                                          round(dealscan_merged['CR_svm'].sum()/1000000000, 3),
                                                          round(dealscan_merged['CR_rf'].sum()/1000000000, 3),
                                                          round(dealscan_merged['CR_gb'].sum()/1000000000, 3)
                                                         ]
Eco_evaluation

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Economic effects (gain/loss)

# COMMAND ----------

len(dealscan_merged)

# COMMAND ----------

len(dealscan_merged.dropna(subset = ['rate']))

# COMMAND ----------

def Eco_loss(df, pred_model, CR_model):
    df_fn = df[(df['default'] == 1) & (df[pred_model] == 0)]
    loss_fn = sum(df_fn['FacilityAmt'] * df_fn['ExchangeRate'])
    df_fp = df[(df['default'] == 0) & (df[pred_model] == 1)]
    loss_fp = sum(df_fp['FacilityAmt'] * df_fp['ExchangeRate'] *df_fp['rate'])
    loss_cr = 0.115 * df[CR_model].sum()
    loss = loss_fn + loss_fp + loss_cr
    return loss

# COMMAND ----------

loss_lr = Eco_loss(dealscan_merged, 'pred_lr', 'CR_lr')
loss_lasso = Eco_loss(dealscan_merged, 'pred_lasso', 'CR_lasso')
loss_svm = Eco_loss(dealscan_merged, 'pred_svm', 'CR_svm')
loss_rf = Eco_loss(dealscan_merged, 'pred_rf', 'CR_rf')
loss_gb = Eco_loss(dealscan_merged, 'pred_gb', 'CR_gb')

# COMMAND ----------

Eco_evaluation.loc['Economical loss (billion)', :] = [round(loss_lr/1000000000, 3),
                                                      round(loss_lasso/1000000000, 3),
                                                      round(loss_svm/1000000000, 3),
                                                      round(loss_rf/1000000000, 3),
                                                      round(loss_gb/1000000000, 3)
                                                     ]
Eco_evaluation

# COMMAND ----------


