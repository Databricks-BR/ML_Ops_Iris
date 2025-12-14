# Databricks notebook source
# This notebook is meant to run batch inference on the top of new iris samples

# COMMAND ----------

# MAGIC %pip install mlflow=='3.4.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from sklearn import datasets
from mlflow.pyfunc import load_model
import pandas as pd
import mlflow
from datetime import datetime

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pedroz_e2edata_dev")
catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

model_name = f'{catalog_name}.default.iris_model'

# COMMAND ----------

# Pull the dataset for running the inference
iris_samples = datasets.load_iris(as_frame=True)
df_samples = pd.DataFrame(data = iris_samples['data'], columns = iris_samples['feature_names'])
df_samples.columns = df_samples.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df_samples.head()

# COMMAND ----------

model_uri = f"models:/{model_name}@champion"
model = load_model(model_uri)

# COMMAND ----------

predictions = model.predict(df_samples)
df_samples['prediction'] = predictions

df_samples.head()

# COMMAND ----------

df_samples['actual_label'] = iris_samples['target']
df_samples.head()

# COMMAND ----------

# Adding the model_id and prediction_timestamp columns to the dataframe - 
# these are required if, in the future, you want to use Lakehouse Monitoring to track the performance of the model
mlflow_client = mlflow.tracking.MlflowClient()
model_version = mlflow_client.get_model_version_by_alias(model_name, "champion").version

df_samples['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
df_samples['model_id'] = model_version

display(df_samples)

# COMMAND ----------

df_spark = spark.createDataFrame(df_samples)

# COMMAND ----------

try:
    display(spark.table(f"{catalog_name}.default.iris_data").limit(5))
    table_exists = True
except:
    table_exists = False

# COMMAND ----------

if table_exists: # append
    df_spark.write.mode("append").saveAsTable(f"{catalog_name}.default.iris_inferences")
else: # create table from scratch
    df_spark.write.mode("overwrite").saveAsTable(f"{catalog_name}.default.iris_inferences")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog_name}.default.iris_inferences LIMIT 5"))

# COMMAND ----------

# Enabling the Change Data Feed is a recommended practice for Inference Monitoring using Lakehouse Monitoring
# When CDF is enabled, only newly appended data is processed. 
spark.sql(f"ALTER TABLE {catalog_name}.default.iris_inferences SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
