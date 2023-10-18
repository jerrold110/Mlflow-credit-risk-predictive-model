#import argparse
import sys
import os
from pathlib import Path

from pandas import read_csv
from numpy import random
from sklearn.model_selection import train_test_split
import mlflow
# add command line arguments
#parser = argparse.ArgumentParser()
#parser.add_argument("data_file", help="Path to the data file to split")
# parse the command-line arguments
#args = parser.parse_args()
data_file = sys.argv[1]
print(data_file)
print(type(data_file))

aaa

# Split the data with a seed
random(1)
df = read_csv(data_file)
train, test = train_test_split(df, test_size=0.2, stratify=df['class'])
# export the data to new csv_files
parent_directory = os.path.normpath(data_file + os.sep + os.pardir)
train_path = os.path.normpath(parent_directory + os.sep + Path('Train.csv'))
train.to_csv(train_path)
