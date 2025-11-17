'''
You've done a great job learning how to derive predictions using a trained model.

In this practice task, you'll complete the code to make predictions and evaluate the model's performance. 
Here are your objectives for this task:

Make predictions with the trained model.
Set the evaluator's parameters correctly.
Calculate the model's accuracy.
Dive into it and enhance your practical skills with PySpark MLlib!
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

# Display the first 5 rows of the predictions DataFrame
predictions.show(5)

# TODO: Configure the appropriate settings to calculate the model's accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# TODO: Compute the accuracy of the model on the test data
accuracy = evaluator.evaluate(predictions)

# Output the calculated accuracy of the model
print("Model Accuracy:", accuracy)

# Stop the Spark session
spark.stop()