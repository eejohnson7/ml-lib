'''
In solution.py, complete the following:

Import the preprocess_data function.
Call preprocess_data with the SparkSession and data path.
'''

from pyspark.sql import SparkSession
# TODO: Import the preprocess_data function from preprocess_data.py
from preprocess_data import preprocess_data

# Initialize a Spark session
spark = SparkSession.builder.appName("PreprocessData").getOrCreate()

# TODO: Call the preprocess passing the SparkSession and data path ("iris.csv")                  
train_data, test_data = preprocess_data(spark, "iris.csv")

# Show the count of rows in the training data
print("Training Data Count:", train_data.count())

# Show the count of rows in the test data
print("Test Data Count:", test_data.count())

# Stop the Spark session
spark.stop()