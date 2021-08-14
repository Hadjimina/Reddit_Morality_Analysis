# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import logging as lg
from datetime import datetime
from multiprocessing import Queue

import coloredlogs
import numpy as np
import pandas as pd
import praw
from tqdm import tqdm

import constants as CS
import parallel_process as p_process
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
            (get_punctuation_count, CS.POST_TEXT)
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

    
    
    feature_df_list = []
    processes = []
    q = Queue()  # Build a single queue to send to all process objects...
    for i in range(0, CS.NR_THREADS):
        sub_post = df_posts_split[i]
        p = p_process.parallel_process(q, i, sub_post, features_to_generate )
        p.start()
        processes.append(p)

    # join all
    [proc.join() for proc in processes]
    
    #put all result in list
    while not q.empty():
        feature_df_list.append(q.get())

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
