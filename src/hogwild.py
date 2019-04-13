import sys
import json
import csv

from os import path
from time import time
from datetime import datetime

import numpy as np
from operator import add
from pyspark.sql import SparkSession

from svm import SVM
from load_data import DataLoader
import settings as s


def fit_then_dump(data, learning_rate, lambda_reg, frac, niter=100):
    start_time = time()
    model = SVM(learning_rate, lambda_reg, frac, s.dim)
    fit_log = model.fit(data.training_set, data.validation_set, niter)
    end_time = time()

    training_accuracy = model.predict(data.training_set)
    validation_accuracy = model.predict(data.validation_set)
    valdiation_loss = model.loss(data.validation_set)
    # Save results in a log
    log = [{'start_time': datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            'running_time': end_time - start_time,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'validation_loss': valdiation_loss,
            'fit_log': fit_log}]

    logname = f'{datetime.utcfromtimestamp(end_time).strftime("%Y%m%d_%H%M%S")}_{learning_rate}_{lambda_reg}_{frac}_log.json'
    with open(path.join(s.logpath, logname), 'w') as outfile:
        json.dump(log, outfile)

    return training_accuracy, validation_accuracy, valdiation_loss


def grid_search(data, learning_rates, lambdas, batch_fracs):
    values = [(
        'learning_rate',
        'lambda_reg',
        'frac',
        'training_accuracy',
        'validation_accuracy',
        'validation_loss'
    )]
    for learning_rate in learning_rates:
        for lambda_reg in lambdas:
            for frac in batch_fracs:
                training_accuracy, validation_accuracy, valdiation_loss = fit_then_dump(
                    data, learning_rate, lambda_reg, frac, niter=100)
                values.append((learning_rate, lambda_reg, frac,
                               training_accuracy, validation_accuracy, valdiation_loss))

    with open(path.join(s.logpath, 'grid_search_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(values)


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Spark")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    data = DataLoader(spark)

    #fit_then_dump(data, s.learning_rate, s.lambda_reg, 0.01, 50)

    learning_rates = np.linspace(0.015, 0.045, 5)
    batch_fracs = [0.005, 0.085, 0.01]
    lambdas = [1e-5, 1e-4, 1e-3]
    grid_search(data, learning_rates, lambdas, batch_fracs)

    spark.stop()
