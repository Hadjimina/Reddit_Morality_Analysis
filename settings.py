"""
Global variables
"""
import praw
import logging as lg
import constants as CS
import pandas as pd

def init():
    global reddit
    reddit = praw.Reddit(
            client_id="ChMem9TZYJif1A",
            client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
            user_agent="android:com.example.myredditapp:v1.2.3"
        )

def load_comments():
    # Load comments csv
    global df_comments
    lg.info("Loading comments: "+CS.COMMENTS_RAW)
    df_comments = pd.read_csv(CS.COMMENTS_RAW, index_col=False)