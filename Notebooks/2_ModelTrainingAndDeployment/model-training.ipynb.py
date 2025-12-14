# Databricks notebook source
# This notebook is meant to train a classification model from the Iris dataset and save it to the UC

# COMMAND ----------

# MAGIC %pip install mlflow=='3.4.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient
import requests
from datetime import datetime

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pedroz_e2edata_dev")
catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

model_name = 'iris_model'

# COMMAND ----------

feature_table_name = f'{catalog_name}.default.iris_data'

# COMMAND ----------

experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/{model_name}_{catalog_name}"

# COMMAND ----------

import mlflow

# Create an MLFlow experiment
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# If you want to autolog the model, use the following command
# Note that some of the auto-logging capabilities were set to false because we are logging some metrics 
mlflow.autolog(log_input_examples=False,log_model_signatures=False,log_models=False,log_datasets=False,)

# Start a training run
with mlflow.start_run() as run:
    # Load data from Unity Catalog table
    df_iris = spark.table(feature_table_name).toPandas()
    features = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
    target = 'species'

    X = df_iris[features]
    y = df_iris[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_f1", f1)

    # Infer model signature
    signature = infer_signature(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        name='model',
        signature=signature,
        input_example=X_train.head()
    )

    # Log input dataset for lineage
    data_source = mlflow.data.load_delta(table_name=feature_table_name)
    mlflow.log_input(data_source, context="training")

# COMMAND ----------

# Out of all runs in the experiment, only register the run with the best selected metric
# Important note: this logic is optional and totally depends on your processes, so feel free to customize it!
# If you want, you can simply register the latest run instead, for example. 

selected_metric = 'test_accuracy'
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=[f"metrics.{selected_metric} DESC"], max_results=1)
best_run_id = runs[0].info.run_id

model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, f"{catalog_name}.default.{model_name}")
client.set_registered_model_alias(name=registered_model.name, alias="challenger", version=registered_model.version)
