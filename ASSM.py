# Databricks notebook source
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark import SparkContext
import pandas as pd
import numpy as np

# Dataset
# https://www.kaggle.com/antmarakis/fake-news-data

# References
# https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
# https://spark.apache.org/docs/latest/ml-features#tf-idf
# https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes

# COMMAND ----------

# 1. Collect fake news sources and pre-process data

# Read in both fake and real datasets
pdf_fake = pd.read_csv('file:/dbfs/FileStore/tables/fakenews/fnn_politics_fake.csv')
pdf_real = pd.read_csv('file:/dbfs/FileStore/tables/fakenews/fnn_politics_real.csv')

# Populate column with appropriate labels
pdf_fake['category'] = "fake"
  
# Populate column with appropriate labels
pdf_real['category'] = "real"
  
# Format dfs, we only need the title and the label
del pdf_fake['id']
del pdf_fake['news_url']
del pdf_fake['tweet_ids']

del pdf_real['id']
del pdf_real['news_url']
del pdf_real['tweet_ids']

# Merge dfs into one
pdfs = [pdf_fake, pdf_real]
pdf = pd.concat(pdfs)
# pdf 

d = spark.createDataFrame(pdf)
display(d)

# Ignore the left most column, there are actually 1056 rows

# COMMAND ----------

# 2. Setup our pipeline

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="title", outputCol="words", pattern="\\W")

# stop words
addStopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers']
stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(addStopWords)

# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# Encodes string column to label indices
labelStringIdx = StringIndexer(inputCol="category", outputCol="label")

p = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectors, labelStringIdx])

# fit the pipeline to training documents
pipelineFit = p.fit(d)
data = pipelineFit.transform(d)
# data.show(5)
display(data)

# COMMAND ----------

# 3. Choose a model and train it with the collected data;

# Split the dataframe
train, test = data.randomSplit([0.8, 0.2])

print("Training set: " + str(train.count()))
print("Testing set: " + str(test.count()))

# 3a. Linear Regression
# If a linear regression model is trained with the elastic net parameter α set to 1, it is equivalent to a Lasso model.
# On the other hand, if α is set to 0, the trained model reduces to a ridge regression model. 
lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0)
model = lr.fit(train)

# 4. Run testing on the model with test dataset.
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

# COMMAND ----------

# 3b. Logist Regression using TF-IDF features
from pyspark.ml.feature import HashingTF, IDF

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) 
pipelineTF = Pipeline(stages=[regexTokenizer, stopWordsRemover, hashingTF, idf, labelStringIdx])

pTF = pipelineTF.fit(d)
dataTF = pTF.transform(d)

(trainTF, testTF) = dataTF.randomSplit([0.7, 0.3])
lrTF = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0)
modelTF = lrTF.fit(trainTF)

# 4. Run testing on the model with test dataset.
predictionsTF = modelTF.transform(testTF)
evaluatorTF = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluatorTF.evaluate(predictionsTF)

# COMMAND ----------

#3c. Naive Bayes
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
modelNB = nb.fit(train)

# 4. Run testing on the model with test dataset.
predictionsNB = modelNB.transform(test)
evaluatorNB = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluatorNB.evaluate(predictionsNB)
