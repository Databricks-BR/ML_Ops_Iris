# Databricks notebook source
# MAGIC %pip install mlflow=='3.4.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow
from sklearn import datasets

# COMMAND ----------

dbutils.widgets.text("model_name", "pedroz_e2edata_dev.default.iris_model")
dbutils.widgets.text("model_version", "1")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

# Pull the dataset for running the inference
iris_samples = datasets.load_iris(as_frame=True)
df_samples = pd.DataFrame(data = iris_samples['data'], columns = iris_samples['feature_names'])
df_samples.columns = df_samples.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df_samples['species'] = iris_samples.target.astype(int)
df_samples.head()

# COMMAND ----------

# REQUIRED: add evaluation dataset and target here
eval_data = df_samples
target = "species"
# REQUIRED: add model type here (e.g. "regressor", "databricks-agent", etc.)
model_type = "classifier"

model_uri = f'models:/{model_name}/{model_version}'
# can also fetch model ID and use that for URI instead as described below

with mlflow.start_run(run_name="evaluation") as run:
  mlflow.models.evaluate(
    model=model_uri,
    data=eval_data,
    targets=target,
    model_type=model_type
  )
