import pandas as pd
import sys 
sys.path.append('..')
import constants as CS
import numpy as np
import os
from os import walk
from pathlib import Path
import logging as lg
import coloredlogs
coloredlogs.install()

def save_df(df, name):
    name = name.replace("posts_cleaned","fixed")
    lg.info("    Saving "+name)
    df.to_csv(CS.dataset_dir+name, index=False)

def remove_unnecessary_columns(df):
    post_cols = list(df.columns)
    post_cols = list(filter(lambda x: "post" in x.split("_") and x != "post_id", post_cols))

    if len(post_cols) > 0:
        lg.warning("    Removing unnecessary columns ({0})".format(str(post_cols)))
    else:
        return None

    for i in post_cols:
        df = df.drop(i, axis=1)
    return df

def do_fix_liwc():
    lg.warning("Checking if there are any shitty LIWC csvs")

    filenames = next(walk(CS.dataset_dir), (None, None, []))[2]  # [] if no file
    filenames = list(filter(lambda x: not "mini" in x and not "Identifier" in x, filenames))
    for name in filenames:

        name_cutoff_pairs = np.array([["LIWC", "WC"], ["moral_foundations", "WC"]])
        names = list(name_cutoff_pairs[:,0])

        valid_name = list(filter(lambda x: x in name, names))
        if len(valid_name) > 0:
            name_idx = names.index(valid_name[0]) 
            cutoff = name_cutoff_pairs[name_idx][1]
            
            lg.info("  Loading "+name)
            df = pd.read_csv(CS.dataset_dir+name)
            
            cols = list(df.columns)
            
            cutoff_idx_in_df = cols.index(cutoff)

            if not "post_id" in cols:
                lg.warning("    This looks like a shitty LIWC csv")
                # Get first row and create correct column names
                fst_row_vals = list(df.iloc[0])
                cols[:cutoff_idx_in_df] = fst_row_vals[:cutoff_idx_in_df]
                
                # drop first column
                df = df.iloc[1:]
                # rename
                cols = list(map(lambda x: ' '.join(x.split()), cols))
                df.columns = cols

                
                df = remove_unnecessary_columns(df)
                save_df(df,name)
                
            else:
                lg.info("    This csv looks fine")
                df = remove_unnecessary_columns(df)
                if not df is None:
                    save_df(df,name)

do_fix_liwc()