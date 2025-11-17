'''
In this task, you'll focus on addressing a critical issue in the process of saving and loading a logistic regression model 
using PySpark.

Tackle this exercise and ensure your model persistence works seamlessly!
'''

from pyspark.sql import SparkSession
from preprocess_data import preprocess_data
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName("ModelSaving").getOrCreate()

# Preprocess the dataset
train_data, test_data = preprocess_data(spark, "iris.csv")

# Initialize the logistic regression model with specified feature and label columns
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the logistic regression model to the training data
lr_model = lr.fit(train_data)

# Save the trained model as "my_model" in the current working directory
lr_model.write().overwrite().save("my_model")

# Load the saved model
loaded_model = LogisticRegressionModel.load("my_model")

# Make predictions on the test set
predictions = loaded_model.transform(test_data)

# Initialize an evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Compute accuracy using the evaluator
accuracy = evaluator.evaluate(predictions)

# Print the accuracy of the loaded model
print("Loaded Model Accuracy:", accuracy)

# Stop the Spark session
spark.stop()