## Introduction
This project attempts to use the Mlflow framework to organise the machine learning life cycle of a credit default risk predictive model. All experiments, models, data, data-splits, and objects are registered in a registry, and pipelines are used for each step in the ml lifecycle to facilitate the standards of MLops of monitoring, deployment, reproducibility, and CI/CD integration. 

The object registry is meant to be accessed with the MLflow ui for interaction and can be stored in a local filesystem or remotely on a server, one only has to specify where. `mlflow ui`


## Sections
Models prototypes are created in `Modelling` with python notebooks.

Model lifecycle workflows are in `model_build_pipeline` and include: data preparation, build, test, deploy. This section is set up as a project and entry points (commands to run) are declared in `MLproject` that control what workflow to execute. The python environment has to be standardised and can be mananged with `conda` or in a docker image, but I am using the local environment in this case. 