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

def minify(remove_extracted=False, file=None):
    dataset_dir = CS.dataset_dir
    filenames = next(walk(dataset_dir), (None, None, []))[2]  # [] if no file
    filenames = list(filter(lambda x: not "mini" in x and not "Identifier" in x, filenames))

    # get posts_cleaned
    posts_cleaned = ""
    for name in filenames:
        split_flag = name.split("_")[0] == "posts"
        posts_cleaned = name if "posts_cleaned" in name and split_flag else posts_cleaned
    posts_min = pd.read_csv(dataset_dir+posts_cleaned, index_col=False)
        
    if remove_extracted:
        lg.warning("Removing extracted")
        if not file:
            lg.error("No file entered")
        lg.info(f"Already extracted {file}")
        extracted_pth = dataset_dir+"extracted/"+file
        posts_extracted = pd.read_csv(extracted_pth)
        post_ids_to_remove = list(posts_extracted["post_id"])
        posts_min = posts_min[~posts_min['post_id'].isin(post_ids_to_remove)]
        lg.info(f"{len(post_ids_to_remove)} posts were already extracted, now we only have {posts_min.shape[0]}")
        
    else:
        lg.warning("Minifying with Frac = "+str(CS.MINIFY_FRAC))

        # minify posts_cleaned
        lg.info("  Minifying "+posts_cleaned)
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
        downsampeled_perc = df_sampeled_size/df_orig_size
        removed = df_orig_size - df_sampeled_size
        if remove_extracted:
            lg.info("    Saving {0}, removed {1}, ratio {2}".format(split_name[len(split_name)-1], removed, downsampeled_perc))
        else:
            lg.info("    Saving {0}, minified to {1}%".format(split_name[len(split_name)-1], downsampeled_perc))
        df_cur.to_csv(min_name, index=False)

        
    #lg.warning("Remember to update the scores if applicable")
        
#if __name__ == "main":
remove_extracted = "-e" in sys.argv or "-extract" in sys.argv
csvs = list(filter(lambda x: ".csv" in x, sys.argv))
csv_file = csvs[0] if len(csvs) > 0 else None
minify(remove_extracted, csv_file)