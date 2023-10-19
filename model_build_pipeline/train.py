import os
import sys
import pandas as pd
import mlflow
from mlflow import log_metric, log_params, log_artifact
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from numpy import squeeze
import joblib
import importlib

data_path = sys.argv[1] 
cv_folds = int(sys.argv[2])
model_type = sys.argv[3]

## get data
train = pd.read_csv(data_path+'/train_fe.csv')
test = pd.read_csv(data_path+'/test_fe.csv')

train_x = train.drop(labels='class', axis=1)
train_y = train[['class']]
test_x = test.drop(labels='class', axis=1)
test_y = test[['class']]

# determine columns to scale
# only scale the numerical columns
numerical_columns = list(train_x.columns[(train_x.dtypes == 'int64') | (train_x.dtypes == 'float64')])

# fit the scaler on the numerical
s = StandardScaler()
s.fit(train[numerical_columns])
# seralize the scaling object in the data_path
joblib.dump(s, data_path+'/scaler.gz')

# transform train_x and test_x
train_x[numerical_columns] = s.transform(train_x[numerical_columns])
test_x[numerical_columns] = s.transform(test_x[numerical_columns])

# train model
def pandas_to_numpy(df):
    return squeeze(df.values)

train_x = pandas_to_numpy(train_x)
train_y = pandas_to_numpy(train_y)
test_x = pandas_to_numpy(test_x)
test_y = pandas_to_numpy(test_y)

def import_module_from_path(file_path):
    module_name = model_type
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    return custom_module

try:
    if model_type == 'random_forest':
        model_path = 'Models/random_forest.py'
    elif model_type == 'logistic_regression':
        model_path = 'Models/logistic_regression.py'
    else:
        raise ValueError
except ValueError:
    print(f'model type input {model_type} is invalid')

with mlflow.start_run() as run:
    module_imported = import_module_from_path(model_path)
    model_grid = module_imported.model(train_x, train_y, cv_folds, 'recall')
    model_be = model_grid.best_estimator_
    #gridsearchcv_results = pd.DataFrame(data=lr_grid.cv_results_, columns=lr_grid.cv_results_.keys())
    #mlflow.log_artifact(gridsearchcv_results)
    best_params = model_grid.best_params_
    log_params(best_params)
    
    # validate model with test data
    y_ = model_be.predict(test_x)
    y_prob = model_be.predict_proba(test_x)
    signature = mlflow.models.infer_signature(test_x, y_)
    mlflow.sklearn.log_model(model_be,
                             artifact_path='sklearn-model',
                             registered_model_name='sklearn-model-'+model_type, 
                             signature=signature)

    # log metrics
    log_metric("accuracy", accuracy_score(test_y, y_))
    log_metric("recall", recall_score(test_y, y_))
    log_metric("f1", f1_score(test_y, y_))
    log_metric("area roc", roc_auc_score(test_y, y_prob[:,1]))

    # log scaler artifact
    log_artifact(data_path+'/scaler.gz')



    

