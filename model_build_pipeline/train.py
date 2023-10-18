import os
import sys
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler

data_path = sys.argv[1] #string
ext_path = sys.argv[2] #string


# start logging data on mlflow

with mlflow.start_run() as run:
    None
# get data

# create model pipeline with scaler and cross-validation

# train model

# validate model with test data

# record metrics