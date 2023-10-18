#import argparse
import sys
import os
from pathlib import Path

from pandas import read_csv
import numpy
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
# add command line arguments

# 
data_file = sys.argv[1] #string
# if os.path.exists(data_file):
# Split the data with a seed
numpy.random.seed(11)
df = read_csv(data_file)
train, test = train_test_split(df, test_size=0.2, stratify=df['class'])

# export the data to new csv_files
parent_directory = os.path.dirname(data_file) # string
train_dir = parent_directory + '/train.csv'
test_dir = parent_directory + '/test.csv'
train.to_csv(train_dir, index=False)
test.to_csv(test_dir, index=False)

# log these as pandas datasets

pd_data1: PandasDataset = mlflow.data.from_pandas(train, source=train_dir)
pd_data2: PandasDataset = mlflow.data.from_pandas(test, source=test_dir)
#pd_data2 = PandasDataset(mlflow.data.from_pandas(test), source=test_dir)

with mlflow.start_run():
    mlflow.log_input(pd_data1, context="train data")
    mlflow.log_input(pd_data2, context="test data")