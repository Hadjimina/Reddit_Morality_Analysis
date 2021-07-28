# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import constants
import pandas as pd
import praw, prawcore
import logging as lg
from datetime import datetime, timezone
import coloredlogs
coloredlogs.install()

from feature_functions.speaker_features import * 
"""
Global variables
"""
reddit = None

def main():
    """Iterate over all posts and create features dataframe"""
    
    #initalize reddit object for PRAW
    global reddit
    reddit = praw.Reddit(
        client_id="ChMem9TZYJif1A",
        client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
        user_agent="android:com.example.myredditapp:v1.2.3"
    )

    #iterate over all cleaned posts:
    df_post_cleaned = pd.read_csv(constants.POSTS_CLEAN, index_col=0).head(5) #TODO: remove "index_col=0"
    for row in df_post_cleaned.itertuples():
        
        index, post_id, post_text, post_title, post_author, _, _, _, = row
        
        #karma = get_post_author_karma(post_author)
        #print(karma)

        #age = get_author_age(post_author)
        #print(age)
        
        activity = get_author_amita_post_activity(post_author)
        print(activity)
        
    
if __name__ == "__main__":
    main()