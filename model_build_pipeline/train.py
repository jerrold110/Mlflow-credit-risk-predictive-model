import os
import sys
import pandas as pd
from Feature_extractor import feature_extractor_pipeline
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# feature extractor
base_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(f"{base_directory}")
data = pd.read_csv('Data/credit.csv')

# start logging data on mlflow
mlflow.set_experiment("experiment alpha")
experiment = mlflow.get_experiment_by_name("experiment alpha")
mlflow.set_tracking_uri("http://192.168.0.1:5000")

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
# extract features
    data_fe = feature_extractor_pipeline(data)

# data split and scale
    x = data_fe.drop(columns='class', inplace=True)
    y = data_fe[['class']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=1)

    print(f"Run ID: {run.info.run_id}")
    




