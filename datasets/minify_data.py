"""" Script to minify the big datasets.
      Dataframes get downsampled to CS.MINIFY_FRAC % of their original size
      We rewrite all already minified dataframes

"""
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

dataset_dir = CS.dataset_dir
filenames = next(walk(dataset_dir), (None, None, []))[2]  # [] if no file
filenames = list(filter(lambda x: not "mini" in x and not "Identifier" in x, filenames))

print("")
lg.warning("Minifying with Frac = "+str(CS.MINIFY_FRAC))

# get posts_cleaned
posts_cleaned = ""
for name in filenames:
    split_flag = name.split("_")[0] == "posts"
    posts_cleaned = name if "posts_cleaned" in name and split_flag else posts_cleaned

# minify posts_cleaned
lg.info("  Minifying "+posts_cleaned)
posts_min = pd.read_csv(dataset_dir+posts_cleaned, index_col=False)
posts_min = posts_min.sample(frac=CS.MINIFY_FRAC)

# save
posts_min_name = "{0}_mini.csv".format(dataset_dir+Path(dataset_dir+posts_cleaned).stem)
lg.info("    Saving {0} minified to {1}%".format(posts_cleaned, CS.MINIFY_FRAC))
posts_min.to_csv(posts_min_name, index=False)

# get ids of posts_cleaned
post_ids = list(posts_min["post_id"])
for name in filenames:
    if name == posts_cleaned or "raw" in name:
        continue

    lg.info("  Minifying "+name)
    df_cur = pd.read_csv(dataset_dir+name, index_col=False)

    if  not "post_id" in list(df_cur.columns):
        lg.error("  This dataframe does not have a post_id column. Skipping")
        continue
    
    # only get rows with post_id of downsampeled posts_cleaned
    df_orig_size = df_cur.shape[0]
    df_cur = df_cur[df_cur["post_id"].isin(post_ids)]
    df_sampeled_size = df_cur.shape[0]

    
    min_name = "{0}_mini.csv".format(dataset_dir+Path(dataset_dir+name).stem)
    split_name = min_name.split("/")
    lg.info("    Saving {0} minified to {1}%".format(split_name[len(split_name)-1], df_sampeled_size/df_orig_size))
    df_cur.to_csv(min_name, index=False)
    