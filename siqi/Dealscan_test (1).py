# Databricks notebook source
# MAGIC %md
# MAGIC ## Testing how many defaults could be linked with observations in Dealscan

# COMMAND ----------

pip install openpyxl

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# load data
df = pd.read_csv("/dbfs/FileStore/Thesis data/df_V2.csv")

# COMMAND ----------

df = df.drop(columns = ['P2'])

# COMMAND ----------

df = df.iloc[:,1:]

# COMMAND ----------

# X = df.drop(['gvkey', 'fyear','datadate', 'default','default_date', 'gsector'], axis = 1)
X = df[['gvkey', 'fyear','datadate', 'default','default_date']]
y = df['default']

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = df['default'], test_size = 0.3, random_state=0)

# COMMAND ----------

df = pd.DataFrame(X_test)
df

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------merging-----------------------------------

# COMMAND ----------

dealscan_link = pd.read_excel("/dbfs/FileStore/Thesis data/raw data/dealscan_link.xlsx")
lender = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/lender.csv")
loan = pd.read_csv("/dbfs/FileStore/Thesis data/raw data/loan.csv")

# COMMAND ----------

loan

# COMMAND ----------

dealscan_link = dealscan_link[['facid', 'bcoid', 'gvkey']]
loan = loan[['FacilityID', 'BorrowerCompanyID', 'FacilityStartDate', 'FacilityAmt', 'Currency', 'ExchangeRate', 'Maturity']]

# COMMAND ----------

dealscan_merged = pd.merge(dealscan_link, loan, how="inner", left_on= ['facid', 'bcoid'] ,right_on=['FacilityID','BorrowerCompanyID'])

# COMMAND ----------

dealscan_merged["FacilityStartDate"] = pd.to_datetime(dealscan_merged["FacilityStartDate"], format='%Y%m%d')
df["datadate"] = pd.to_datetime(df["datadate"], format='%Y-%m-%d')

# COMMAND ----------

dealscan_merged

# COMMAND ----------

dealscan_merged[['fyear', 'datadate', 'default', 'default_date']] = ''


# COMMAND ----------

for i in range(len(dealscan_merged)):
    gvkey = dealscan_merged.iloc[i]["gvkey"]
    date = dealscan_merged.iloc[i]["FacilityStartDate"]
    temp_df = df[(df["gvkey"] == gvkey) & (df["datadate"] < date ) & (df["datadate"] + pd.offsets.DateOffset(years=1) > date)]
    if len(temp_df) == 0:
        continue
    else:
        dealscan_merged.iloc[i, -4:] = temp_df.iloc[0, 1:]


        
#dealscan_merged.iloc[i, -4] = df[(df["gvkey"] == gvkey) & (df["datadate"] < date ) & (df["datadate"] + pd.offsets.DateOffset(years=1) > date)].iloc[:, 1:]

# COMMAND ----------

dealscan_merged

# COMMAND ----------

dealscan_merged= dealscan_merged.drop(index = dealscan_merged.index[dealscan_merged['default'] == ''])
dealscan_merged

# COMMAND ----------

dealscan_merged['default'].sum()

# COMMAND ----------

dealscan_merged[dealscan_merged['default'] == 1]

# COMMAND ----------

dealscan_merged[(dealscan_merged['default'] == 1) & (dealscan_merged['default_date'] > dealscan_merged['FacilityStartDate'] )]

# COMMAND ----------

#-------------------------------break line--------------------------------

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


