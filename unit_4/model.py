'''
Congratulations on reaching the final task of the course! 🎉 You've come a long way in mastering model persistence. 
As a culminating challenge, tackle saving, loading and verifying a model using PySpark.

Complete the code snippet below:

Save the trained model with a specified name.
Load the saved model successfully.
Make predictions on the test set using the loaded model.
Showcase your skills and bring it all together with accuracy and efficiency. You've got this!
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

# TODO: Save the trained model as "my_model" in the current working directory
lr_model.write().overwrite().save("my_model")

# TODO: Load the saved model
loaded_model = LogisticRegressionModel.load("my_model")

# TODO: Make predictions on the test set using the loaded model
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