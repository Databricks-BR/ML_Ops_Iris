# Databricks notebook source
# MAGIC %pip install mlflow=='3.4.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "pedroz_e2edata_dev.default.iris_model")
dbutils.widgets.text("model_version", "1")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
  ServedEntityInput,
  EndpointCoreConfigInput
)
from databricks.sdk.errors import ResourceDoesNotExist
from mlflow import MlflowClient

# COMMAND ----------

# Promote the model version to Champion

client = MlflowClient()

client.set_registered_model_alias(
    f'{model_name}', 
    "Champion", 
    model_version
)

# COMMAND ----------

# Create a serving endpoint for the model

# REQUIRED: Enter serving endpoint name
serving_endpoint_name = model_name.replace('.', '-') + "-endpoint"

w = WorkspaceClient()  # Assumes DATABRICKS_HOST and DATABRICKS_TOKEN are set
served_entities=[
  ServedEntityInput(
    entity_name=model_name,
    entity_version=model_version,
    workload_size="Small",
    scale_to_zero_enabled=True
  )
]

# Update serving endpoint if it already exists, otherwise create the serving endpoint
try:
  w.serving_endpoints.update_config(name=serving_endpoint_name, served_entities=served_entities)
except ResourceDoesNotExist:
  w.serving_endpoints.create(name=serving_endpoint_name, config=EndpointCoreConfigInput(served_entities=served_entities))
