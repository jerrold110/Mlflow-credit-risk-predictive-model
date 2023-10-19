import mlflow
import os

if __name__ == "__main__":
    work_dir = os.getcwd()
    mlflow.projects.run(uri=work_dir,
                        entry_point='train',
                        parameters={'data_path':'Data_preprocessed/feature_extractor_1',
                                    'cv_folds':10,
                                    'model':'random_forest'},
                        experiment_name = 'Test',
                        env_manager='local')

"""
    # split
    mlflow.projects.run(uri=work_dir,
                        entry_point='split_data',
                        parameters={'data_file':'Data/credit_1/credit.csv'},
                        experiment_name = 'Test',
                        env_manager='local')

    # extract features for a given model
    mlflow.projects.run(uri=work_dir,
                        entry_point='extract_data',
                        parameters={'data_path':'Data/credit_1',
                                    'extractor_file':'Data_preprocessed/feature_extractor_1/extractor.py'},
                        experiment_name = 'Test',
                        env_manager='local') 

    # train and validate
    mlflow.projects.run(uri=work_dir,
                        entry_point='train',
                        parameters={'data_path':'Data_preprocessed/feature_extractor_1',
                                    'cv_folds':5,
                                    'model':'random_forest'},
                        experiment_name = 'Test',
                        env_manager='local')
"""  