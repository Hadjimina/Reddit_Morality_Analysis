# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import logging as lg
import queue
import sys
from datetime import datetime
from multiprocessing import Queue, process, set_start_method
from os import walk

import coloredlogs
import numpy as np
import pandas as pd
import praw

import constants as CS
import helpers.df_visualisation as vis
import helpers.globals_loader as globals_loader
import parallel_process as p_process

from helpers.helper_functions import *
from helpers.process_helper import *

coloredlogs.install()


def main():
    """Iterate over all posts and create features dataframe"""
    df_posts = globals_loader.df_posts
    df_posts_split = np.array_split(df_posts, CS.NR_THREADS)

    # Do multiprocessing
    processes = []
    q = Queue() 
    
    for i in range(0, CS.NR_THREADS):
        sub_post = df_posts_split[i]
        #p = p_process.parallel_process(q, i, sub_post, features_to_generate, globals_loader.stz_nlp )
        p = p_process.parallel_process(q, i, sub_post, CS.FEATURES_TO_GENERATE_MP)
        p.start()
        processes.append(p)

    # Consume queue content as it comes (avoids deadlock)
    multi_feature_df_list = []
    while True:
        try:
            result = q.get(False, 0.01)
            multi_feature_df_list.append(result)
        except queue.Empty:
            pass
        allExited = True
        for t in processes:
            if t.exitcode is None:
                allExited = False
                break
        if allExited & q.empty():
            break
    
    feature_df_multi = pd.concat(multi_feature_df_list, axis=0, join="inner")  
    lg.info("Finished Multiprocessing")
    
    # Do mono processing
    mono_feat_df_list = process_run(CS.FEATURES_TO_GENERATE_MONO, df_posts, CS.MONO_ID)
    feature_df_mono = pd.concat(mono_feat_df_list, axis=1, join="inner")  
    lg.info("Finished Monoprocessing")

    # Merge mono and multiprocessing
    feature_df = feature_df_multi.merge(feature_df_mono, left_on="post_id", right_on="post_id", validate="1:1")
    
    # Merge generate Features with LIWC & Moral foundations
    # TODO: Should we change how duplicates are handeled? right now we have suffixes
    if CS.LOAD_LIWC:
        feature_df = feature_df.merge(globals_loader.df_liwc, left_on="post_id", right_on=CS.LIWC_PREFIX+"post_id", validate="1:1") 
    if CS.LOAD_FOUNDATIONS:
        feature_df = feature_df.merge(globals_loader.df_foundations, left_on="post_id", right_on=CS.FOUNDATIONS_PREFIX+"post_id", validate="1:1") 
    # Create histogram and sample texts as png
    vis.generate_report(feature_df)

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y")
    feature_df = feature_df.drop("post_text",axis=1)
    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    feature_df.to_csv(CS.OUTPUT_DIR+"features_output_"+date_time+mini+".csv", index=False)
    
    
if __name__ == "__main__":
    
    if "-vis" in sys.argv:
        
        filenames = next(walk(CS.OUTPUT_DIR), (None, None, []))[2]  # [] if no file
        filenames = list(filter(lambda x: ".csv" in x, filenames))
        filenames = sorted(filenames, reverse=True)

        lg.info("Only generating report from "+filenames[0])
        df = pd.read_csv(CS.OUTPUT_DIR+filenames[0], index_col=False)

        globals_loader.load_posts()
        df_posts = globals_loader.df_posts[["post_id", "post_text"]]
        df = pd.merge(df, df_posts, on=['post_id','post_id'])
        
        vis.generate_report(df)
    else:
        globals_loader.init()  
        lg_str = "Using {0} threads".format(CS.NR_THREADS)
        if CS.NR_THREADS < 2:
            lg.warning(lg_str[:-1]+"!")
        else:
            lg.info(lg_str)
        main()
