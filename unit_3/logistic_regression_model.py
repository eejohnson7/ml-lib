'''
You've made significant progress in learning how to derive predictions using a trained model. Now it's time to put that 
knowledge into practice!

In this task, you will complete the code to:

Make predictions using the trained logistic regression model.
Correctly set the parameters for the evaluator.
Calculate and display the model's accuracy.
Tackle this challenge to enhance your skills in using PySpark MLlib.
'''

from pyspark.sql import SparkSession
from preprocess_data import preprocess_data
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName("ModelEvaluation").getOrCreate()

# Preprocess the dataset
train_data, test_data = preprocess_data(spark, "iris.csv")

# Initialize the logistic regression model with specified feature and label columns
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the logistic regression model to the training data
lr_model = lr.fit(train_data)

# TODO: Make predictions on the test set
predictions = lr_model.transform(test_data)

# TODO: Initialize a MulticlassClassificationEvaluator to calculate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# TODO: Compute the accuracy of the model on the test data
accuracy = evaluator.evaluate(predictions)

# TODO: Display the calculated accuracy of the model
print("Model Accuracy:", accuracy)

# Stop the Spark session
spark.stop()