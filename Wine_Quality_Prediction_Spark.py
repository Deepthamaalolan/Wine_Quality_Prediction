import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.tree import RandomForestModel
from pyspark.sql.functions import col

def clean_data(df): 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))
"""main function for application"""
if __name__ == "__main__":
    print("Starting Spark Application")

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    
    # create spark context to report logging information related spark
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    print("Configuring Hadoop for S3")
    # spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
    # spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.sparkContext._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    input_path = "s3://wimequalitypredictiondataset/ValidationDataset.csv"
    valid_path = "s3://wimequalitypredictiondataset/ValidationDataset.csv"
    model_path="s3://wimequalitypredictiondataset/trainedmodel"

# read csv file in DataFram 
    # print(f"Reading CSV file from {input_path}")
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load("./ValidationDataset.csv"))
    
    df1 = clean_data(df)
    all_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'chlorides',
                        'total sulfur dioxide',
                        'density',
                        'sulphates',
                        'alcohol',
                    ]
    

rf = PipelineModel.load("./rf/")
    
predictions = rf.transform(df1)
results = predictions.select(['prediction', 'label'])

print("Displaying the first 5 predictions")
print(predictions.show(5))

evaluator = MulticlassClassificationEvaluator(
                                            labelCol='label', 
                                            predictionCol='prediction', 
                                            metricName='accuracy')
# printing accuracy of above model
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy of wine prediction model = ', accuracy)
metrics = MulticlassMetrics(results.rdd.map(tuple))
print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())

print("Exiting Spark Application")
sys.exit(0)
