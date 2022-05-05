# Databricks notebook source
import pandas as pd

# COMMAND ----------

#using pandas
pandas_df = pd.read_csv("/dbfs/FileStore/siqi_data/addresses.csv", header='infer')
pandas_df.head()

# COMMAND ----------

# or using spark
sparkDF = spark.read.csv("/FileStore/siqi_data/addresses.csv", header="true", inferSchema="true")
sparkDF.show()

# COMMAND ----------



# COMMAND ----------


