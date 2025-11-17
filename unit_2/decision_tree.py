'''
In this task, you'll switch from using a logistic regression model to a decision tree classifier within PySpark's MLlib.

Here's what you'll need to update:

Import DecisionTreeClassifier instead of LogisticRegression.
Change the initialization of LogisticRegression to DecisionTreeClassifier.
Fit the DecisionTreeClassifier model to the training data instead of LogisticRegression.
Print the properties of the DecisionTreeClassifier, specifically the depth of the tree using model.depth and the 
number of leaves using model.numNodes.
This will help solidify your knowledge about using different classification models. You're making great progress; keep it up!
'''

from pyspark.sql import SparkSession
from preprocess_data import preprocess_data
# TODO: Import the DecisionTreeClassifier instead of LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier

# Initialize a Spark session
spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

# Preprocess the dataset
train_data, test_data = preprocess_data(spark, "iris.csv")

# TODO: Change the model
# - Replace LogisticRegression with DecisionTreeClassifier
# - Update the variable name to indicate the model type (e.g., `lr` to `dt`)
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# TODO: Fit the DecisionTreeClassifier instance to the training data
model = dt.fit(train_data)

# TODO: Print depth of the decision tree, use model.depth
print("Depth of decision tree:\n", model.depth)

# TODO: Print number of leaves in the decision tree, use model.numNodes
print("Number of leaves in decision tree:\n", model.numNodes)

# Stop the Spark session
spark.stop()