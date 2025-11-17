'''
Now, let's switch gears and explore different model evaluation metrics.

Your task is to change the model evaluation metric from "accuracy" to "f1", reflecting a balance between precision and 
recall. This change will help you see the impact of using a different metric on model evaluation.

Give it a try and see how this metric alters your view of model performance.
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

# TODO: Change the metric used to evaluate the model from "accuracy" to "f1"
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# TODO: Rename the variable to f1_score
f1_score = evaluator.evaluate(predictions)

# TODO: Print the F1 Score
print("F1 Score:", f1_score)

# Stop the Spark session
spark.stop()