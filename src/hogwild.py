from __future__ import print_function

import sys
import numpy as np
from scipy.sparse import csr_matrix
from os import path

from operator import add
from pyspark.sql import SparkSession

from svm import SVM
import settings as s

def line_to_topic(r):
    topic, doc_id, _ = r[0].strip().split(' ')
    return int(doc_id), set([topic])

def line_to_features(r):
    r = r[0].strip().split(' ')
    features = [feature.split(':') for feature in r[2:]]
    col_idx = np.array([0] + [int(idx) + 1 for idx, _ in features])
    row_idx = np.array([0]*(len(features) + 1))
    data = np.array([1.] + [float(value) for _, value in features])
    return int(r[0]), csr_matrix((data, (row_idx, col_idx)), shape=(1, s.dim))

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Test App")\
        .getOrCreate()

    topics_lines = spark.read.text(
        path.join(s.path, 'datasets/rcv1-v2.topics.qrels')).rdd.map(line_to_topic)
    doc_category = topics_lines.reduceByKey(lambda x, y: x | y).map(
        lambda x: (x[0], 1) if 'CCAT' in x[1] else (x[0], -1)).persist()

    train_lines = spark.read.text(path.join(s.path, 'datasets/lyrl2004_vectors_train.dat')).rdd.map(line_to_features)

    """test_lines = []
    for i in range(4):
        test_lines.append(spark.read.text(path.join(s.path, 'datasets/lyrl2004_vectors_test_pt'+str(i)+'.dat')))
    test_lines = test_lines[0].union(test_lines[1]).union(test_lines[2]).union(test_lines[3])"""

    training_set, validation_set = train_lines.join(doc_category).map(lambda x : (x[1][0], x[1][1])).randomSplit([1 - s.validation_frac, s.validation_frac], seed=1)
    training_set.persist()
    validation_set.persist()
    """test_set = test_lines.rdd.map(line_to_features).join(doc_category).map(lambda x : (x[1][0], x[1][1])).persist()"""

    model = SVM(s.learning_rate, s.lambda_reg, s.dim)
    model.fit(training_set, 100)
    print('accuracy', model.predict(training_set))

    spark.stop()
