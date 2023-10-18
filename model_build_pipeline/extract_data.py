import sys
import os

from pandas import read_csv
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.data.pandas_dataset import PandasDataset

data_path = sys.argv[1] #string
ext_path = sys.argv[2] #string

import importlib.util
import sys

def import_module_from_path(file_path):
    module_name = "Feature extractor"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    return custom_module

# import extractor module
extractor = import_module_from_path(ext_path)
print(help(extractor.fe_pipeline))

# extract features from train and test data
train = read_csv(data_path + '/train.csv')
test = read_csv(data_path + '/test.csv')
train_fe = extractor.fe_pipeline(train)
test_fe = extractor.fe_pipeline(test)

# export data into a subdirectory of the extractor object
parent_directory = os.path.dirname(ext_path) # string
train_fe_dir = parent_directory + '/train_fe.csv'
test_fe_dir = parent_directory + '/test_fe.csv'
train_fe.to_csv(train_fe_dir, index=False)
test_fe.to_csv(test_fe_dir, index=False)

# log these datasets
pd_data1: PandasDataset = mlflow.data.from_pandas(train_fe, source=train_fe_dir)
pd_data2: PandasDataset = mlflow.data.from_pandas(test_fe, source=test_fe_dir)

with mlflow.start_run():
    mlflow.log_input(pd_data1, context="train data features")
    mlflow.log_input(pd_data2, context="test data features")
