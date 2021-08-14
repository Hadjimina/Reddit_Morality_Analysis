# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import logging as lg
import threading
from datetime import datetime

import coloredlogs
import numpy as np
import pandas as pd
import praw
from tqdm import tqdm

import constants as CS
import feature_thread as f_thread
import globals_loader
import helpers.df_visualisation as vis
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from helpers.helper_functions import *

coloredlogs.install()

def generate_report(df):
    """Generate report by visualising the columns of the dataframe as 
        histograms and listing 3 example post (low, medium, high value)
        below the histograms. Saves as report.png in home directory of script

    Parameters
    ----------
    df : Dataframe
        Dataframe with format of feature_to_df

    """
    lg.info("Generating report")
    vis.df_to_plots(df)
    vis.df_to_text_png(df)


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
            #(get_punctuation_count, CS.POST_TEXT)
        ],
        "behaviour":[
            
        ],
        "reactions":[
            (get_judgement_labels, CS.POST_ID)
        ]
    }
    
    setup_load_dfs()
    df_posts = globals_loader.df_posts

    df_posts_split = np.array_split(df_posts, CS.NR_THREADS)

    threadLock = threading.Lock()
    threads = []
    feature_df_list = []

    # Split up & start threading
    for i in range(len(df_posts_split)):
        sub_post = df_posts_split[i]
        threads.append(f_thread.feature_thread(i, sub_post, features_to_generate))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        feature_df_list.append(t.df)

    feature_df = pd.concat(feature_df_list, axis=0, join="inner")       

    # Create histogram and sample texts as png
    generate_report(feature_df)

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y")
    feature_df = feature_df.drop("post_text",axis=1)
    feature_df.to_csv(CS.OUTPUT_DIR+"features_output_"+date_time+".csv")
    
    
if __name__ == "__main__":
    globals_loader.init()  
    main()
