'''
Here's what you need to do in the preprocess_data.py file:

Use StringIndexer to convert the 'species' column into numerical labels in a new 'label' column.
Combine numerical features using VectorAssembler into a 'features' vector.
Properly split the data with randomSplit().
'''

from pyspark.ml.feature import StringIndexer, VectorAssembler

def preprocess_data(spark, data_path):
    # Load the dataset
    raw_data = spark.read.csv(data_path, header=True, inferSchema=True)

    # TODO: Complete the StringIndexer to use 'species' as input and 'label' as output
    indexer = StringIndexer(inputCol="species", outputCol="label")
        
    # TODO: Complete the transformation to add the 'label' column to indexed_data
    indexed_data = indexer.fit(raw_data).transform(raw_data)

    # TODO: Complete the VectorAssembler to combine specified feature columns into 'features'
    # Use "sepal_length", "sepal_width", "petal_length", "petal_width"
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )

    # TODO: Complete the transformation to add the 'features' column to vectorized_data
    vectorized_data = assembler.transform(indexed_data)
    
    # TODO: Select 'features' and 'label' columns to create final_data
    final_data = vectorized_data.select("features", "label")

    # TODO: Use randomSplit to divide final_data into train_data and test_data
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    return train_data, test_data