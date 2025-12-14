# Databricks notebook source
# This notebook is meant to show a simple example of how to use the near real-time inference endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# COMMAND ----------

w = WorkspaceClient()

sample_input = [
    {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2
    }
]

prediction = w.serving_endpoints.query(
    name="pedroz_e2edata_dev-default-iris_model-endpoint",
    dataframe_records=sample_input
)

display(prediction)

# COMMAND ----------

print(
    'Input:',
    sample_input,
    '\nOutput: ',
    prediction.predictions
)
