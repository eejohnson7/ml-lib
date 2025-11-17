'''
In this task, you'll fix bugs in a PySpark program designed to train and evaluate a logistic regression model.

Here’s your chance to ensure the code correctly trains the model, makes predictions and calculates its accuracy. 
Tackle these bugs and enhance your troubleshooting skills even further!
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

# Make predictions on the test set
predictions = lr_model.transform(test_data)

# Display the first 5 rows of the predictions DataFrame
predictions.show(5)

# Initialize the evaluator with the desired settings
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Compute the accuracy of the model on the test data
accuracy = evaluator.evaluate(predictions)

# Output the calculated accuracy of the model
print("Model Accuracy:", accuracy)

# Stop the Spark session
spark.stop()