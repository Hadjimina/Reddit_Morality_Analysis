"""
Global variables
"""
import praw
import logging as lg
import constants as CS
import pandas as pd
import spacy
from spacy.language import Language
from spacytextblob.spacytextblob import SpacyTextBlob

def init():
    if CS.USE_MINIFIED_DATA:
        lg.warning("Using minified data (fraction: {0})".format(CS.MINIFY_FRAC))
    if not hasattr(globals(), 'reddit'):
        load_reddit_settings()
        
    if not hasattr(globals(), 'df_posts'):
        load_posts()

    if not hasattr(globals(), 'df_comments'):
        if CS.LOAD_COMMENTS:
            load_comments()
        else:
            lg.warning("Skipping loading comments...")

    if not hasattr(globals(), 'df_liwc'):
        if CS.LOAD_LIWC:
            load_liwc()
        else:
            lg.warning("Skipping loading liwc...")

    if not hasattr(globals(), 'df_foundations'):
        if CS.LOAD_FOUNDATIONS:
            load_foundations()
        else:
            lg.warning("Skipping loading foundations...")

    if not hasattr(globals(), 'nlp'):
        load_spacy()
    

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
        lg.info("Nr. of posts {0}".format(df_posts.shape[0]))

    # Drop "Unnamed" cols if they exist
    for c in list(df_posts.columns):
        if "Unnamed" in c:
            df_posts = df_posts.drop(c, axis=1)

def load_foundations():
    # Load foundations csv
    global df_foundations
    lg.info("Loading foundations: "+CS.FOUNDATIONS)
    df_foundations = pd.read_csv(CS.FOUNDATIONS, index_col=False)

def load_liwc():
    # Load foundations csv
    global df_liwc
    lg.info("Loading foundations: "+CS.LIWC)
    df_liwc = pd.read_csv(CS.LIWC, index_col=False)

def load_spacy():
    global nlp
    lg.info("Loading spacy")
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("spacytextblob")
