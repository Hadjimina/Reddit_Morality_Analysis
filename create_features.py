# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/


import logging as lg
import queue
import sys
from datetime import datetime
from multiprocessing import Queue, process
from os import walk

import coloredlogs
import numpy as np
import pandas as pd
import praw
from tqdm import tqdm

import constants as CS
import helpers.df_visualisation as vis
import helpers.globals_loader as globals_loader
import parallel_process as p_process
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from helpers.helper_functions import *

coloredlogs.install()


def main():
    """Iterate over all posts and create features dataframe"""
    
    tqdm.pandas()

    #initalize reddit object for PRAW
    global reddit
    reddit = praw.Reddit(
        client_id="ChMem9TZYJif1A",
        client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
        user_agent="android:com.example.myredditapp:v1.2.3"
    )

    features_to_generate = {
        "speaker":[
            #(get_author_amita_post_activity, CS.POST_AUTHOR),
            #(get_author_age, CS.POST_AUTHOR),
            #(get_post_author_karma, CS.POST_AUTHOR)
        ], 
        "writing_sty":[
            (get_punctuation_count, CS.POST_TEXT)
        ],
        "behaviour":[
            
        ],
        "reactions":[
            (get_judgement_labels, CS.POST_ID)
        ]
    }

    df_posts = globals_loader.df_posts
    df_posts_split = np.array_split(df_posts, CS.NR_THREADS)

    
    feature_df_list = []
    processes = []
    q = Queue() 
    
    for i in range(0, CS.NR_THREADS):
        sub_post = df_posts_split[i]
        p = p_process.parallel_process(q, i, sub_post, features_to_generate )
        p.start()
        processes.append(p)

    # Consume queue content as it comes (avoids deadlock)
    feature_df_list = []
    while True:
        try:
            result = q.get(False, 0.01)
            feature_df_list.append(result)
        except queue.Empty:
            pass
        allExited = True
        for t in processes:
            if t.exitcode is None:
                allExited = False
                break
        if allExited & q.empty():
            break

    feature_df = pd.concat(feature_df_list, axis=0, join="inner")       

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
        main()
