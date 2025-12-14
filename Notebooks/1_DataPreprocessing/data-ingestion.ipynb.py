# Databricks notebook source
# This notebook is meant to extract the data from sklearn.datasets and ingest it into a table in the UC

# COMMAND ----------

from sklearn import datasets 
import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mhiltner_dev")
catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

iris_data = datasets.load_iris(as_frame=True)
df_iris = pd.DataFrame(data = iris_data['data'], columns = iris_data['feature_names'])
df_iris.columns = df_iris.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df_iris['species'] = iris_data['target']
df_iris.head()

# COMMAND ----------

df_iris['id'] = range(1, len(df_iris) + 1)
df_iris.head()

# COMMAND ----------

spark_df_iris = spark.createDataFrame(df_iris)

# COMMAND ----------

try:
    display(spark.table(f"{catalog_name}.default.iris_data").limit(5))
    table_exists = True
except:
    table_exists = False

# COMMAND ----------

if not table_exists:
    spark_df_iris.write.mode("overwrite").saveAsTable(f"{catalog_name}.default.iris_data")
    spark.sql(f"ALTER TABLE {catalog_name}.default.iris_data ALTER COLUMN id SET NOT NULL")
    spark.sql(f"ALTER TABLE {catalog_name}.default.iris_data ADD CONSTRAINT pk_id PRIMARY KEY (id)")
    print("Created table and added a primary key to it")
else:
    spark_df_iris.write.mode("overwrite").saveAsTable(f"{catalog_name}.default.iris_data")
    print("Overwrote table data")

# COMMAND ----------

df = spark.sql(f"SELECT * FROM {catalog_name}.default.iris_data")
display(df.limit(5))
