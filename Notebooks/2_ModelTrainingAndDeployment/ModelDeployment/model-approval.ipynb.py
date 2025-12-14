# Databricks notebook source
# MAGIC %pip install mlflow=='3.4.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "pedroz_e2edata_dev.default.iris_model")
dbutils.widgets.text("model_version", "1")
dbutils.widgets.text("approval_tag_name", "approval")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
approval_tag_name = dbutils.widgets.get("approval_tag_name")

# COMMAND ----------

from mlflow import MlflowClient

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")

# fetch the model version's UC tags
model_tags = client.get_model_version(model_name, model_version).tags

# check if any tag matches the approval tag name
if not any(tag == approval_tag_name for tag in model_tags.keys()):
  raise Exception("Model version not approved for deployment")
else:
  # if tag is found, check if it is approved
  if model_tags.get(approval_tag_name).lower() == "approved":
    print("Model version approved for deployment")
  else:
    raise Exception("Model version not approved for deployment")
