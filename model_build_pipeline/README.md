## Usage
The whole lifecycle is organised as a project in `MLproject`. This file declares the entry points into the project. Each entry point contains a set of instructions that performs the steps in a pipeline (whether it is data preparation, or model train, or model deploy) and is executed through the command-line-interfact or the python API. I use the python API in this case, because there are many parameters to pass, so using the python API makes this process more convenient and manageable. https://mlflow.org/docs/latest/projects.html#running-projects. The execution commands are in the file `run.py`, and the steps are declared in `MLproject`.

To execute each workflow. Configure the entrypoints in MLproject, then set up the command and parameters in run.py and execute the operations in the entry-point with `python run.py`. If unspecified in the default entry point is `main`. As I am storing the registy in the local filesystem of this repo, the URI parameter of mlflow.projects.run() is the current working directory, but this can be changed to a remote server.

https://mlflow.org/docs/latest/projects.html#project-environments
https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run


## Contents
`Data` contains the raw data which has not been split or processed, and comes straight from the internal datawarehouse. The subfolder `credit_1` contains a specific version of the data from the warehouse, and two files train and test which is a random 80-20 split of the data for training and testing, generated by a fixed seed. The untouched, and seed-split data are logged in the registry and are reproducible and traceable. 

`Data_preprocessed` contains data that has undergone feature selection and extraction from the split data in `data` and is traceable to the original files before any processing in `Data` through logging in the data registry, accessible from the mlflow ui. It also contains the training-data-fitted scaling objects and the unique feature extractor code so that data, proprocessing objects, and feature extractors are coupled during versioning and the whole process of data management is traceable. 

`Models` contains the untrained model and its hyperparameter cross-validation grid search framework objects from sklearn. These files are meant to be called in `train.py` where information of the specific model to be trained is passed in the entry point parameters in `run.py`. Currently only code for logistic regression and random forest models are present. It can be easily extended for further models in sklearn and other deep learning packages like TF or PT. Every ML model train, package, evaluation, and validation step is registered in the model registry.

## Workflows and features
Workflows:
* Data split and versioning workflow
* Feature extraction workflow
* Model train and validate workflow
* Deployment workflow

Mlflow features:
* Project framework orchestrated in `MLproject`
* Experiment management
* Tracking of all data, data manipulators, model/preprocessing objects
* Model and artifact registry
* Model packaging (object, metadata, experiment tracking, versioning)
