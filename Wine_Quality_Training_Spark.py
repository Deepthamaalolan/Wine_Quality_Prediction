import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_data(df):
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    print("Starting Spark Application")


    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")


    input_path = "s3://wimequalitypredictiondataset/TrainingDataset.csv"
    valid_path = "s3://wimequalitypredictiondataset/ValidationDataset.csv"
    output_path = "s3://wimequalitypredictiondataset/trainedmodel"

    print(f"Reading training CSV file from {input_path}")
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(input_path))
    
    train_data_set = clean_data(df)

    print(f"Reading validation CSV file from {valid_path}")
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(valid_path))
    
    valid_data_set = clean_data(df)

    all_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol', 'quality']

    print("Creating VectorAssembler")
    assembler = VectorAssembler(inputCols=all_features, outputCol='features')
    
    print("Creating StringIndexer")
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    print("Caching data for faster access")
    train_data_set.cache()
    valid_data_set.cache()
    
    print("Creating RandomForestClassifier")
    rf = RandomForestClassifier(labelCol='label', 
                                featuresCol='features',
                                numTrees=150,
                                maxDepth=15,
                                seed=150,
                                impurity='gini')
    
    print("Creating Pipeline for training")
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_data_set)

    print("Transforming data using the trained model")
    predictions = model.transform(valid_data_set)

    print("Evaluating the trained model on the validation set")
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                  predictionCol='prediction', 
                                                  metricName='accuracy')

   
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model = {accuracy}')
    
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print(f'Weighted f1 score of wine prediction model = {metrics.weightedFMeasure()}')

    print("Retraining model on multiple parameters using CrossValidator")
    cvmodel = None
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [6, 9]) \
        .addGrid(rf.numTrees, [50, 150]) \
        .addGrid(rf.minInstancesPerNode, [6]) \
        .addGrid(rf.seed, [100, 200]) \
        .addGrid(rf.impurity, ["entropy", "gini"]) \
        .build()
    
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=2)

    print("Fitting CrossValidator to the training data")
    cvmodel = crossval.fit(train_data_set)
    
    print("Saving the best model to new param `model`")
    model = cvmodel.bestModel
    

    predictions = model.transform(valid_data_set)
    results = predictions.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model (after CrossValidation) = {accuracy}')
    
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print(f'Weighted f1 score of wine prediction model (after CrossValidation) = {metrics.weightedFMeasure()}')

    print("Saving the best model to S3")
    model_path = output_path
    model.write().overwrite().save(model_path)
    

    sys.exit(0)
