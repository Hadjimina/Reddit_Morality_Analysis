import os
from os import walk
import pandas as pd
from pathlib import Path

dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/data/"
filenames = next(walk(dataset_dir), (None, None, []))[2]  # [] if no file

for name in filenames:
    df_min = pd.read_csv(dataset_dir+name, index_col=False).head(10)
    min_name = "{0}_mini.csv".format(dataset_dir+Path(dataset_dir+name).stem)
    df_min.to_csv(min_name, index=False)