import os
import mlflow

if __name__ == "__main__":
    None


import mlflow
from pathlib import Path
import os
# working directory cannot be in the same location as this file for some reason
work_dir = os.getcwd()

mlflow.projects.run(uri=work_dir,
                    entry_point='split_data',
                    parameters={'data_file':'Data/credit_1/credit.csv'},
                    experiment_name = 'Test',
                    env_manager='local') # 'conda' if using conda for env management


mlflow.projects.run(uri=work_dir,
                    entry_point='extract_data',
                    parameters={'data_path':'Data/credit_1',
                                'extractor_file':'Data_preprocessed/feature_extractor_1/extractor.py'},
                    experiment_name = 'Test',
                    env_manager='local') # 'conda' if using conda for env management
