"""
Global variables
"""
import praw
import logging as lg
import constants as CS
import pandas as pd
import spacy
import sys
from spacytextblob.spacytextblob import SpacyTextBlob

def init():
    """Load all required global variables such as dataframes.
        Setup the post title either as a standalone feature that is treated the same way as post text or the post title is prepended ot the post text
    """

    if CS.USE_MINIFIED_DATA:
        lg.warning("Using minified data (fraction: {0})".format(CS.MINIFY_FRAC))
    if not hasattr(globals(), 'reddit'):
        load_reddit_settings()
        
    if not hasattr(globals(), 'df_posts'):
        if CS.LOAD_POSTS:
            load_posts()    
        else:
            lg.warn("Skipping loading posts...")

    if not hasattr(globals(), 'df_comments'):
        if CS.LOAD_COMMENTS:
            load_comments()
        else:
            lg.warning("Skipping loading comments...")

    if not hasattr(globals(), 'df_liwc'):
        if CS.LOAD_LIWC:
            load_liwc()
            if CS.TITLE_AS_STANDALONE:
                load_liwc_title()
        else:
            lg.warning("Skipping loading liwc...")

    if not hasattr(globals(), 'df_foundations'):
        if CS.LOAD_FOUNDATIONS:
            load_foundations()
            if CS.TITLE_AS_STANDALONE:
                load_foundations_title()
        else:
            lg.warning("Skipping loading foundations...")

    if not hasattr(globals(), 'nlp'):
        if  len([item for sublist in CS.FEATURES_TO_GENERATE_MONO["writing_sty"] for item in sublist]) > 0 or CS.DO_TOPIC_MODELLING:
            load_spacy()

    if CS.TITLE_AS_STANDALONE:
        add_standalone_title_features()
        
    

def load_reddit_settings():
    """Load the reddit settings
    """
    global reddit
    reddit = praw.Reddit(
            client_id="ChMem9TZYJif1A",
            client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
            user_agent="android:com.example.myredditapp:v1.2.3"
        )

def load_comments():
    """Load the comments csv
    """
    global df_comments
    lg.info("Loading comments: "+CS.COMMENTS_RAW)
    df_comments = pd.read_csv(CS.COMMENTS_RAW, index_col=False)


def load_posts():
    """Load the posts csv
    """
    global df_posts
    lg.info("Loading posts: "+CS.POSTS_CLEAN)
    df_posts = pd.read_csv(CS.POSTS_CLEAN, index_col=False)
    if CS.USE_MINIFIED_DATA:
        lg.info("Nr. of posts {0}".format(df_posts.shape[0]))

    # Drop "Unnamed" cols if they exist
    for c in list(df_posts.columns):
        if "Unnamed" in c:
            df_posts = df_posts.drop(c, axis=1)

    if not CS.TITLE_AS_STANDALONE:
        # Here we modify the post_title column, but since we do not use post_title on its own in any feature this is ok
        # => we want to have a space between the end of the post title and the start of the post text if the title is standalone (?)
        df_titles = df_posts["post_title"].str.strip()
        bool_dot = df_titles.str.endswith(".", na=False)
        bool_q = df_titles.str.endswith("?", na=False)
        bool_exc = df_titles.str.endswith("!", na=False)
        bool_merged = bool_dot | bool_q | bool_exc

        df_titles[~bool_merged] = df_titles[~bool_merged]+"."

        df_posts["post_text"] = df_titles+" "+df_posts["post_text"]


def load_foundations():
    """Load the moral foundations csv for the posts only
    """
    global df_foundations
    lg.info("Loading foundations: "+CS.FOUNDATIONS)
    df_foundations = pd.read_csv(CS.FOUNDATIONS, index_col=False)
    df_foundations = df_foundations.add_prefix(CS.FOUNDATIONS_PREFIX)

def load_foundations_title():
    """Load the moral foundations csv for only the titles
    """
    global df_foundations_title
    lg.info("Loading foundations: "+CS.FOUNDATIONS_TITLE)
    df_foundations_title = pd.read_csv(CS.FOUNDATIONS_TITLE, index_col=False)
    df_foundations_title = df_foundations_title.add_prefix(CS.FOUNDATIONS_TITLE_PREFIX)

def load_liwc():
    """ Load the liwc csv for the posts only
    """
    global df_liwc
    lg.info("Loading liwc: "+CS.LIWC)
    df_liwc = pd.read_csv(CS.LIWC, index_col=False)
    df_liwc = df_liwc.add_prefix(CS.LIWC_PREFIX)

def load_liwc_title():
    """Load the liwc csv for the titles only
    """
    global df_liwc_title
    lg.info("Loading liwc: "+CS.LIWC_TITLE)
    df_liwc_title = pd.read_csv(CS.LIWC_TITLE, index_col=False)
    df_liwc_title = df_liwc_title.add_prefix(CS.LIWC_TITLE_PREFIX)

def load_spacy():
    """Setup the spacy pipeline
    """
    global nlp
    lg.info("Loading spacy")
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("spacytextblob")


def add_standalone_title_features():
    """ Expand the feature dictionaries to apply every function that is applied on the post text, also on the post title
    """

    feature_dicts = [CS.FEATURES_TO_GENERATE_MP, CS.FEATURES_TO_GENERATE_MONO]
    for i in range(len(feature_dicts)):
        feature_dict = feature_dicts[i]
        feature_dict_copy = feature_dict.copy()
        for k, v in feature_dict.items():
            v_copy = v.copy()
            for tpl in v:
                fn = tpl[0]
                idx = tpl[1]
                if idx == CS.POST_TEXT:
                    v_copy.append((fn, CS.POST_TITLE))
            feature_dict_copy[k] = v_copy
        if i == 0:
            CS.FEATURES_TO_GENERATE_MP = feature_dict_copy
        elif i == 1:
            CS.FEATURES_TO_GENERATE_MONO = feature_dict_copy

