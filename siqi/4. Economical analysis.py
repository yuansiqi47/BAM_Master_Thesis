# Databricks notebook source
# MAGIC %md
# MAGIC # Economical analysis

# COMMAND ----------

import pandas as pd
import numpy as np
import math
from scipy.stats import norm

# COMMAND ----------

# load data
df_pd = pd.read_csv("/dbfs/FileStore/Siqi thesis/df_pd.csv")

# COMMAND ----------

df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merging datasets together

# COMMAND ----------

dealscan_link = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/dealscan_link.csv")
loan = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/loan.csv")
interest = pd.read_csv("/dbfs/FileStore/Siqi thesis/raw data/interest.csv")

# COMMAND ----------

loan.head()

# COMMAND ----------

# MAGIC %md
# MAGIC merging dealscan data with link data

# COMMAND ----------

# get only columns only needed
dealscan_link = dealscan_link[['facid', 'bcoid', 'gvkey']]
loan = loan[['FacilityID', 'BorrowerCompanyID', 'FacilityStartDate', 'FacilityAmt', 'Currency', 'ExchangeRate', 'Maturity', 'LoanType']]
# merge data together
dealscan_merged = pd.merge(dealscan_link, loan, how="inner", left_on= ['facid', 'bcoid'] ,right_on=['FacilityID','BorrowerCompanyID'])
# change type of date data to date type
dealscan_merged["FacilityStartDate"] = pd.to_datetime(dealscan_merged["FacilityStartDate"], format='%Y%m%d')
df_pd["datadate"] = pd.to_datetime(df_pd["datadate"], format='%Y-%m-%d')
# drop unnecessary columns
dealscan_merged = dealscan_merged.drop(columns = ['facid','bcoid'])

# COMMAND ----------

# MAGIC %md
# MAGIC merging dealscan data with PD data

# COMMAND ----------

# create empty columns
dealscan_merged[['fyear', 'datadate', 'default', 'default_date', 'gsector', 'pred_lr', 'pd_lr', 
                 'pred_lasso', 'pd_lasso', 'pred_svm', 'pd_svm', 'pred_rf', 'pd_rf', 
                 'pred_gb', 'pd_gb']] = ''

# COMMAND ----------

# merge dealscan data with the PD data
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

# drop unmatched observations
dealscan_merged = dealscan_merged.drop(index = dealscan_merged.index[dealscan_merged['default'] == ''])

# COMMAND ----------

dealscan_merged.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC Merging with interest data

# COMMAND ----------

interest.head()

# COMMAND ----------

# make alteration to the interest by combing the base rate and fixed rate together
interest = interest[interest['BaseRate'].isin(['LIBOR','Fixed Rate'])]
interest = interest.drop(index = interest.index[interest.duplicated(['FacilityID'])]).reset_index(drop = True)
interest['rate'] = float()
#  0.0093457 is the annual LIBOR rate on Feb 01, 2022
interest.loc[interest['BaseRate'] == 'LIBOR', 'rate']= (interest['MinBps'] + interest['MaxBps'])/2/10000 + 0.0093457
interest.loc[interest['BaseRate'] == 'Fixed Rate', 'rate'] = (interest['MinBps'] + interest['MaxBps'])/2/10000 
interest.head()

# COMMAND ----------

# merging the loan data with the interest data
dealscan_merged = pd.merge(dealscan_merged, interest, how="left",on='FacilityID')
# drop the observations with NA in rates and maturity
dealscan_merged = dealscan_merged.dropna(subset = ['rate', 'Maturity']).reset_index(drop = True)
dealscan_merged.head()

# COMMAND ----------

print("Number of the loans ", len(dealscan_merged)) 
print("Number of the actual defaulted loans ", dealscan_merged['default'].sum()) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Capital Requirement calculation

# COMMAND ----------

# assign the LGD value to the loan observations based on the report: 
# https://www.spglobal.com/marketintelligence/en/news-insights/blog/corporate-credit-risk-trends-in-developing-markets-a-loss-given-default-lgd-perspective
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
    '''a function to calculate the capital requirement'''
    # convert M from month to year
    M = M/12    
    # as PD smaller than 0.00000293 results in a negaitve value of risk weight through the formula,
    # set the capital requirement to be 0.
    if PD < 0.00000293:
        return 0
    # since in our algorithmn, we don't issue loans to applications with estimated PD larger than 0.5,
    # set the capital requirement to be 0.
    elif PD >= 0.5:
        return 0
    else:
        R = 0.12*(1-math.exp(-50*PD))/(1-math.exp(-50))+0.24* (1 - (1-math.exp(-50*PD))/(1-math.exp(-50)))
        b = (0.11852-0.05478*np.log(PD))**2
        RW = (LGD*norm.cdf(1/np.sqrt(1-R)*norm.ppf(PD)+np.sqrt(R/(1-R))*norm.ppf(0.999))-LGD*PD)*(1+(M-2.5)*b)/(1-1.5*b)*1.25*1.06
        CR = RW * loan_amt * exchange_rate * 0.08
        return CR

# COMMAND ----------

# obtain lists of LGD, M, loan value and exchange rate
LGD = dealscan_merged['LGD'].tolist()
M = dealscan_merged['Maturity'].tolist()
loan_amt = dealscan_merged['FacilityAmt'].tolist()
exchange_rate = dealscan_merged['ExchangeRate'].tolist()

# COMMAND ----------

# get capital requirements based each observation
dealscan_merged['CR_lr'] = list(map(Captial_Requirement, dealscan_merged['pd_lr'].tolist(), LGD, M, loan_amt, exchange_rate))
dealscan_merged['CR_lasso'] = list(map(Captial_Requirement, dealscan_merged['pd_lasso'].tolist(), LGD, M, loan_amt, exchange_rate))
dealscan_merged['CR_svm'] = list(map(Captial_Requirement, dealscan_merged['pd_svm'].tolist(), LGD, M, loan_amt, exchange_rate))
dealscan_merged['CR_rf'] = list(map(Captial_Requirement, dealscan_merged['pd_rf'].tolist(), LGD, M, loan_amt, exchange_rate))
dealscan_merged['CR_gb'] = list(map(Captial_Requirement, dealscan_merged['pd_gb'].tolist(), LGD, M, loan_amt, exchange_rate))

# COMMAND ----------

# create a table to show the figures
model = ['Logistic regression (baseline)', 'Logistic regression (Lasso)', 'SVM', 'Random Forest', 'Gradient Boosting']
CR_evaluation = pd.DataFrame(
    columns = ['Loan rate', 'Capital requirement', 'Savings', 'Percent'],
    index = model)

# COMMAND ----------

CR_evaluation.loc[:, 'Loan rate'] = [round(sum(dealscan_merged['pred_lr'] == 0)/len(dealscan_merged),3),
                                     round(sum(dealscan_merged['pred_lasso'] == 0)/len(dealscan_merged), 3),
                                     round(sum(dealscan_merged['pred_svm'] == 0)/len(dealscan_merged), 3),
                                     round(sum(dealscan_merged['pred_rf'] == 0)/len(dealscan_merged), 3),
                                     round(sum(dealscan_merged['pred_gb'] == 0)/len(dealscan_merged), 3)]
CR_evaluation.loc[:, 'Capital requirement'] = [round(dealscan_merged['CR_lr'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_lasso'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_svm'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_rf'].sum()/1000000000, 3),
                                                round(dealscan_merged['CR_gb'].sum()/1000000000, 3)
                                              ]
CR_evaluation.loc[:, 'Savings'] = CR_evaluation.iloc[0]['Capital requirement'] - CR_evaluation.loc[:, 'Capital requirement'] 
CR_evaluation.loc[:, 'Percent'] = round(CR_evaluation.loc[:, 'Savings']/CR_evaluation.iloc[0]['Capital requirement']*100, 1)

# COMMAND ----------

CR_evaluation

# COMMAND ----------

print(CR_evaluation.to_latex())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Economic effects (gain/loss)

# COMMAND ----------

def Eco_loss(df, pred, CR):
    '''a function to calculate the economical loss based on each model'''
    df_fn = df[(df['default'] == 1) & (df[pred] == 0)]
    loss_fn = sum(df_fn['FacilityAmt'] * df_fn['ExchangeRate']* df_fn['LGD'])
    df_fp = df[(df['default'] == 0) & (df[pred] == 1)]
    loss_fp = sum(df_fp['FacilityAmt'] * df_fp['ExchangeRate'] *df_fp['rate'])
    loss_cr = 0.115 * df[CR].sum()
    loss = loss_fn + loss_fp + loss_cr
    return round(loss_fn/1000000000, 3), round(loss_fp/1000000000, 3), round(loss_cr/1000000000, 3), round(loss/1000000000, 3)

# COMMAND ----------

# get the economical loss based on the selected 5 models
lr_loss = list(Eco_loss(dealscan_merged, 'pred_lr', 'CR_lr'))
lasso_loss = list(Eco_loss(dealscan_merged, 'pred_lasso', 'CR_lasso'))
svm_loss = list(Eco_loss(dealscan_merged, 'pred_svm', 'CR_svm'))
rf_loss = list(Eco_loss(dealscan_merged, 'pred_rf', 'CR_rf'))
gb_loss = list(Eco_loss(dealscan_merged, 'pred_gb', 'CR_gb'))

# COMMAND ----------

# create a table to show the figures
products_list = [lr_loss, lasso_loss, svm_loss, rf_loss, gb_loss]

Eco_evaluation = pd.DataFrame (products_list, columns = ['Loss_II', 'Loss_I', 'Loss_cr', 'Loss_total'], 
                   index = ['Logistic regression (baseline)','Logistic regression (Lasso)','SVM','Random Forest','Gradient Boosting'])


# COMMAND ----------

Eco_evaluation.loc[:, 'Gains'] = Eco_evaluation.iloc[0]['Loss_total'] - Eco_evaluation.loc[:, 'Loss_total']
Eco_evaluation.loc[:, 'Percent'] = round(Eco_evaluation.loc[:, 'Gains']/Eco_evaluation.iloc[0]['Loss_total']*100, 1)
Eco_evaluation

# COMMAND ----------

print(Eco_evaluation.to_latex())

# COMMAND ----------


