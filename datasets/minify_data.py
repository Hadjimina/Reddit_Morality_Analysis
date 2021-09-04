import logging as lg
import os
from os import walk
from pathlib import Path
import sys 
sys.path.append('..')

import coloredlogs
import pandas as pd
import constants as CS

coloredlogs.install()

"""" Script to minify the big datasets.
      Dataframes get downsampled to CS.MINIFY_FRAC % of their original size
      comments and already minified dataframes are skipped
"""

dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/data/"
filenames = next(walk(dataset_dir), (None, None, []))[2]  # [] if no file

for name in filenames:
    if "mini" in name or "comment" in name:
        continue

    lg.info("Minifying "+dataset_dir+name)
    df_min = pd.read_csv(dataset_dir+name, index_col=False)
    df_min = df_min.sample(frac=CS.MINIFY_FRAC)

    min_name = "{0}_mini.csv".format(dataset_dir+Path(dataset_dir+name).stem)
    df_min.to_csv(min_name, index=False)
