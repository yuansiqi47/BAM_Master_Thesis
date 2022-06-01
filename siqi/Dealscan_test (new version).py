# Databricks notebook source
# MAGIC %md
# MAGIC ## Economical analysis

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# load data
df_pd = pd.read_csv("/dbfs/FileStore/Siqi thesis/df_pd.csv")

# COMMAND ----------

df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------merging-----------------------------------

# COMMAND ----------

dealscan_link = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/dealscan_link.csv")
loan = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/loan.csv")
interest = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/interest.csv")

# COMMAND ----------

loan.head()

# COMMAND ----------

dealscan_link = dealscan_link[['facid', 'bcoid', 'gvkey']]
loan = loan[['FacilityID', 'BorrowerCompanyID', 'FacilityStartDate', 'FacilityAmt', 'Currency', 'ExchangeRate', 'Maturity', 'LoanType']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_link, loan, how="inner", left_on= ['facid', 'bcoid'] ,right_on=['FacilityID','BorrowerCompanyID'])

# COMMAND ----------

dealscan_merged["FacilityStartDate"] = pd.to_datetime(dealscan_merged["FacilityStartDate"], format='%Y%m%d')
df_pd["datadate"] = pd.to_datetime(df_pd["datadate"], format='%Y-%m-%d')

# COMMAND ----------

dealscan_merged = dealscan_merged.drop(columns = ['facid','bcoid'])

# COMMAND ----------

dealscan_merged[['fyear', 'datadate', 'default', 'default_date', 'gsector', 'pred_lr', 'pd_lr', 
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
        dealscan_merged.iloc[i, -15:] = temp_df.iloc[0, 1:]

        
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

#interest = interest.dropna(subset = ['BaseRate'])
interest = interest[interest['BaseRate'] == 'LIBOR'].reset_index(drop = True)
interest['rate'] = (interest['MinBps'] + interest['MaxBps'])/2/10000 + 0.0092729
#  0.0092729 is the monthly LIBOR rate on 18-05-2022
#interest = interest.groupby('FacilityID').mean()[['rate']]
interest

# COMMAND ----------



# COMMAND ----------

dealscan_merged = pd.merge(dealscan_merged, interest, how="left",on='FacilityID')
dealscan_merged.head()

# COMMAND ----------

dealscan_merged = dealscan_merged.dropna(subset = ['rate', 'Maturity']).reset_index(drop = True)

# COMMAND ----------

dealscan_merged.head()

# COMMAND ----------

len(dealscan_merged)

# COMMAND ----------

dealscan_merged['default'].sum()

# COMMAND ----------

len(dealscan_merged[(dealscan_merged['default'] == 1) & (dealscan_merged['default_date'] > dealscan_merged['FacilityStartDate'] )])

# COMMAND ----------

dealscan_merged.groupby('LoanType').count()

# COMMAND ----------

dealscan_merged.to_csv('/dbfs/FileStore/Siqi thesis/dealscan_merged.csv', index=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### CR calculation

# COMMAND ----------

import pandas as pd
# load data
dealscan_merged = pd.read_csv("/dbfs/FileStore/Siqi thesis/dealscan_merged.csv")

# COMMAND ----------

import numpy as np
import math
from scipy.stats import norm

# COMMAND ----------

dealscan_merged['LGD'] = int()
dealscan_merged.loc[dealscan_merged['gsector']== 10, 'LGD'] = 0.59
dealscan_merged.loc[dealscan_merged['gsector']== 15, 'LGD'] = 0.41
dealscan_merged.loc[dealscan_merged['gsector']== 20, 'LGD'] = 0.38
dealscan_merged.loc[dealscan_merged['gsector']== 25, 'LGD'] = 0.40
dealscan_merged.loc[dealscan_merged['gsector']== 30, 'LGD'] = 0.32
dealscan_merged.loc[dealscan_merged['gsector']== 35, 'LGD'] = 0.28
dealscan_merged.loc[dealscan_merged['gsector']== 45, 'LGD'] = 0.32
dealscan_merged.loc[dealscan_merged['gsector']== 50, 'LGD'] = 0.34


# COMMAND ----------

def Captial_Requirement(PD, LGD, M, loan_amt, exchange_rate):
    # convert M from month to year
    M = M/12    
    if PD < 0.00000293:
        return 0
    elif PD >= 0.5:
        return 0
    else:
        # convert PD to the scheme
        PD = PD/50*28
        R = 0.12*(1-math.exp(-50*PD))/(1-math.exp(-50))+0.24* (1 - (1-math.exp(-50*PD))/(1-math.exp(-50)))
        b = (0.11852-0.05478*np.log(PD))**2
        RW = (LGD*norm.cdf(1/np.sqrt(1-R)*norm.ppf(PD)+np.sqrt(R/(1-R))*norm.ppf(0.999))-LGD*PD)*(1+(M-2.5)*b)/(1-1.5*b)*1.25*1.06
        CR = RW * loan_amt * exchange_rate * 0.08
        return CR

# COMMAND ----------

LGD = dealscan_merged['LGD'].tolist()
M = dealscan_merged['Maturity'].tolist()
loan_amt = dealscan_merged['FacilityAmt'].tolist()
exchange_rate = dealscan_merged['ExchangeRate'].tolist()

# COMMAND ----------

CR_lr = list(map(Captial_Requirement, dealscan_merged['pd_lr'].tolist(), LGD, M, loan_amt, exchange_rate))
CR_lasso = list(map(Captial_Requirement, dealscan_merged['pd_lasso'].tolist(), LGD, M, loan_amt, exchange_rate))
CR_svm = list(map(Captial_Requirement, dealscan_merged['pd_svm'].tolist(), LGD, M, loan_amt, exchange_rate))
CR_rf = list(map(Captial_Requirement, dealscan_merged['pd_rf'].tolist(), LGD, M, loan_amt, exchange_rate))
CR_gb = list(map(Captial_Requirement, dealscan_merged['pd_gb'].tolist(), LGD, M, loan_amt, exchange_rate))

# COMMAND ----------

dealscan_merged['CR_lr'] = CR_lr
dealscan_merged['CR_lasso'] = CR_lasso
dealscan_merged['CR_svm'] = CR_svm
dealscan_merged['CR_rf'] = CR_rf
dealscan_merged['CR_gb'] = CR_gb

# COMMAND ----------

model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
CR_evaluation = pd.DataFrame(
    columns = ['Capital requirement'],
    index = model)

# COMMAND ----------

CR_evaluation.loc[:, 'Capital requirement'] = [round(dealscan_merged['CR_lr'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_lasso'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_svm'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_rf'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_gb'].sum()/1000000000, 3)
                                              ]

# COMMAND ----------

CR_evaluation.loc[:, 'differences'] =CR_evaluation.loc[:, 'Capital requirement'] - CR_evaluation.iloc[0]['Capital requirement']
CR_evaluation

# COMMAND ----------

print(CR_evaluation.to_latex())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Economic effects (gain/loss)

# COMMAND ----------

def Eco_loss(df, pred, CR):
    df_fn = df[(df['default'] == 1) & (df[pred] == 0)]
    loss_fn = sum(df_fn['FacilityAmt'] * df_fn['ExchangeRate']* df_fn['LGD'])
    df_fp = df[(df['default'] == 0) & (df[pred] == 1)]
    loss_fp = sum(df_fp['FacilityAmt'] * df_fp['ExchangeRate'] *df_fp['rate'])
    loss_cr = 0.115 * df[CR].sum()
    loss = loss_fn + loss_fp + loss_cr
    return round(loss_fn/1000000000, 3), round(loss_fp/1000000000, 3), round(loss_cr/1000000000, 3), round(loss/1000000000, 3)

# COMMAND ----------

lr_loss = list(Eco_loss(dealscan_merged, 'pred_lr', 'CR_lr'))
lasso_loss = list(Eco_loss(dealscan_merged, 'pred_lasso', 'CR_lasso'))
svm_loss = list(Eco_loss(dealscan_merged, 'pred_svm', 'CR_svm'))
rf_loss = list(Eco_loss(dealscan_merged, 'pred_rf', 'CR_rf'))
gb_loss = list(Eco_loss(dealscan_merged, 'pred_gb', 'CR_gb'))

# COMMAND ----------

products_list = [lr_loss, lasso_loss, svm_loss, rf_loss, gb_loss]

Eco_evaluation = pd.DataFrame (products_list, columns = ['loss_II', 'loss_I', 'loss_cr', 'loss_total'], 
                   index = ['Logistic regression (baseline)','Logistic regression (Lasso)','SVM','Random Forest','Gradient Boosting'])


# COMMAND ----------

Eco_evaluation.loc[:, 'economical gains'] = Eco_evaluation.iloc[0]['loss_total'] - Eco_evaluation.loc[:, 'loss_total']
Eco_evaluation

# COMMAND ----------

print(Eco_evaluation.to_latex())

# COMMAND ----------


