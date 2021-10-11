# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import logging as lg
import queue
import sys
from multiprocessing import Queue, set_start_method
from os import walk

import coloredlogs
import numpy as np
import pandas as pd
from better_profanity import profanity

import constants as CS
import helpers.df_visualisation as vis
import helpers.globals_loader as globals_loader
import parallel_process as p_process
from feature_functions.topic_modeling import *
from helpers.clean_text import *
from helpers.helper_functions import *
from helpers.process_helper import *
from helpers.requirements import *

coloredlogs.install()

def main():
    """ Iterate over all posts and create features dataframe 
    
    """
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
    
    # fix "uniquely valued index object"
    for df in multi_feature_df_list:
        df.reset_index(inplace=True, drop=True)
    feature_df_multi = pd.concat(multi_feature_df_list, axis=0, join="inner")  
    lg.info("Finished Multiprocessing")
    
    # Do mono processing
    mono_feat_df_list = process_run(CS.FEATURES_TO_GENERATE_MONO, df_posts, CS.MONO_ID)
    feature_df_mono = pd.concat(mono_feat_df_list, axis=1, join="inner")  
    lg.info("Finished Monoprocessing")

    # Merge mono, multiprocessing and topics
    feature_df = feature_df_multi.merge(feature_df_mono, left_on="post_id", right_on="post_id", validate="1:1")

    # Do topic modeling & merge
    if CS.DO_TOPIC_MODELING:
        posts_raw = df_posts["post_text"].to_list()
        post_ids = df_posts["post_id"].to_list()
        topic_df = topic_modeling(posts_raw, post_ids)   
        feature_df = topic_df.merge(feature_df, left_on="post_id", right_on="post_id", validate="1:1")

    # Merge generate features with LIWC & Moral foundations
    if CS.LOAD_LIWC:
        feature_df = feature_df.merge(globals_loader.df_liwc, left_on="post_id", right_on=CS.LIWC_PREFIX+"post_id", validate="1:1") 
        if CS.TITLE_AS_STANDALONE:
            feature_df = feature_df.merge(globals_loader.df_liwc_title, left_on="post_id", right_on=CS.LIWC_TITLE_PREFIX+"post_id", validate="1:1") 
    if CS.LOAD_FOUNDATIONS:
        feature_df = feature_df.merge(globals_loader.df_foundations, left_on="post_id", right_on=CS.FOUNDATIONS_PREFIX+"post_id", validate="1:1") 
        if CS.TITLE_AS_STANDALONE:
            feature_df = feature_df.merge(globals_loader.df_foundations_title, left_on="post_id", right_on=CS.FOUNDATIONS_TITLE+"post_id", validate="1:1") 

    
    # Save features in one big dataframe
    date_time = get_date_str()
    feature_df_to_save = feature_df.drop("post_text",axis=1)
    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    feature_df_to_save.to_csv(CS.OUTPUT_DIR+"features_output_"+date_time+mini+".csv", index=False)

    # Enforce minimum post requiremnets
    if CS.ENFORCE_POST_REQUIREMENTS:
        feature_df = enforce_requirements(feature_df)

    # Create histogram and sample texts as png
    vis.generate_report(feature_df)

    
if __name__ == "__main__":
    
    if "-vis" in sys.argv:
        
        filenames = next(walk(CS.OUTPUT_DIR), (None, None, []))[2]  # [] if no file
        filenames = list(filter(lambda x: ".csv" in x and "features_output" in x, filenames))
        filenames = sorted(filenames, reverse=True)

        lg.info("Only generating report from "+filenames[0])
        df = pd.read_csv(CS.OUTPUT_DIR+filenames[0], index_col=False)
        globals_loader.load_posts()
        df_posts = globals_loader.df_posts[["post_id", "post_text"]]
        df = pd.merge(df, df_posts, on=['post_id','post_id'])

        # Enforce minimum post requiremnets
        if CS.ENFORCE_POST_REQUIREMENTS:
            df = enforce_requirements(df)

        vis.generate_report(df)
    else:
        globals_loader.init()  
        profanity.load_censor_words()

        lg_str = "Using {0} threads".format(CS.NR_THREADS)
        if CS.NR_THREADS < 2:
            lg.warning(lg_str[:-1]+"!")
        else:
            lg.info(lg_str)
        main()
