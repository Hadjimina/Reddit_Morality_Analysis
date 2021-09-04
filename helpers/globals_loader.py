"""
Global variables
"""
import praw
import logging as lg
import constants as CS
import pandas as pd

def init():
    if not hasattr(globals(), 'reddit'):
        load_reddit_settings()
        
    if not hasattr(globals(), 'df_posts'):
        load_posts()

    if not hasattr(globals(), 'df_comments'):
        if CS.LOAD_COMMENTS:
            load_comments()
        else:
            lg.warning("Skipping loading comments...")

    

def load_reddit_settings():
    # setup reddit settings
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


def load_posts():
    # Load comments csv
    global df_posts
    lg.info("Loading posts: "+CS.POSTS_CLEAN)
    df_posts = pd.read_csv(CS.POSTS_CLEAN, index_col=False)
    if CS.USE_MINIFIED_DATA:
        lg.warning("Using minified post data (fraction: {0}, Nr. posts: {1})".format(CS.MINIFY_FRAC, df_posts.shape[0]))


    