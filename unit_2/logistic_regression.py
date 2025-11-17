'''
Your next task is to complete the code to ensure the logistic regression model functions correctly. Here’s what you need to accomplish:

Initialize the LogisticRegression model with the correct feature and label columns.
Fit the model to the training data and store the model to a variable.
Display the coefficient matrix from the trained model.
Display the intercept vector from the trained model.
'''

from pyspark.sql import SparkSession
from preprocess_data import preprocess_data
from pyspark.ml.classification import LogisticRegression

# Initialize a Spark session
spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

# Preprocess the dataset
train_data, test_data = preprocess_data(spark, "iris.csv")

# TODO: Initialize the LogisticRegression model with the appropriate feature and label columns
lr = LogisticRegression(featuresCol="features", labelCol="label")

# TODO: Fit the logistic regression model to the training data and store the model
lr_model = lr.fit(train_data)

# TODO: Display the coefficientMatrix from the trained model
print("CoefficientMatrix:\n", lr_model.coefficientMatrix)

# TODO: Display the interceptVector from the trained model
print("InterceptVector:\n", lr_model.interceptVector)

# Stop the Spark session
spark.stop()