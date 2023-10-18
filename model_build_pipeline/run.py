import mlflow
from pathlib import Path
import os
# working directory cannot be in the same location as this file for some reason
work_dir = os.getcwd()

mlflow.projects.run(uri=work_dir,
                    entry_point='split_data',
                    parameters={'data_file':'Data/credit_1/credit.csv'},
                    experiment_name = 'test')
