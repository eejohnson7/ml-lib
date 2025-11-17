'''
Nice work on mastering model persistence in PySpark! Now, let's test your skills by filling in the blanks to save and load a 
logistic regression model using PySpark MLlib.

Your task is to complete the code snippet below:

Use the correct method to save the trained model with the any model name.
Load the saved model using the appropriate method and name.
Challenge yourself, and let's see how accurately you can apply your learning!


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

# TODO: Save the trained model with a given model name
lr_model.write().overwrite().save("My Model")

# TODO: Load the saved model with the specified name
loaded_model = LogisticRegressionModel.load("My Model")

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