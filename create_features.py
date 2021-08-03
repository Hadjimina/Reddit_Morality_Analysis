# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import constants as CS
import pandas as pd
import praw, prawcore
import logging as lg
from datetime import datetime, timezone
import coloredlogs
coloredlogs.install()

from feature_functions.speaker_features import * 
from feature_functions.writing_style_features import *
"""
Global variables
"""
reddit = None
def feature_to_df(category, ret):

    if  not isinstance(ret, list):
        ret = [ret]

    columns = [k for k,v in ret]
    values = [v for k,v in ret]
    feature_df = pd.DataFrame(values, columns=columns)
    return feature_df

def main():
    """Iterate over all posts and create features dataframe"""
    
    #initalize reddit object for PRAW
    global reddit
    reddit = praw.Reddit(
        client_id="ChMem9TZYJif1A",
        client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
        user_agent="android:com.example.myredditapp:v1.2.3"
    )

    features_to_generate = {
        "speaker":[
            (get_author_amita_post_activity, CS.POST_AUTHOR),
            (get_author_age, CS.POST_AUTHOR),
            (get_post_author_karma, CS.POST_AUTHOR)
        ], 
        "writing_sty":[
            (get_punctuation_count, CS.POST_TEXT)
        ],
        "behaviour":[

        ],
        "reactions":[

        ]
    }
    #iterate over all cleaned posts:
    df_post_cleaned = pd.read_csv(CS.POSTS_CLEAN, index_col=0).head(3) #TODO: remove "index_col=0" & use entire dataset
    for row in df_post_cleaned.itertuples():
        
        index, post_id, post_text, post_title, post_author, _, _, _, = row
        
        for category in features_to_generate:
            for function_tuple in features_to_generate[category]:
                print(category)
                print(function_tuple)
                feature = function_tuple[0](row[function_tuple[1]])
                #print(feature_to_df(category, feature))
                #TODO: iterate columns wise???
                
        #karma = get_post_author_karma(post_author)
        #print(karma)

        #age = get_author_age(post_author)
        #print(age)
        
        #activity = get_author_amita_post_activity(post_author)
        #print(activity)

        

        
    
if __name__ == "__main__":
    settings.init()  
    main()