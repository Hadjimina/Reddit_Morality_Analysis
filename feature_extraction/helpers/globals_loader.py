"""
Global variables
"""
import praw
import logging as lg
import constants as CS
import pandas as pd
import spacy
import sys
import json
from os.path import exists
from spacytextblob.spacytextblob import SpacyTextBlob


def init(do_load_spacy=True, single_post=False):
    """Load all required global variables such as dataframes.
        Setup the post title either as a standalone feature that is treated the same way as post text or the post title is prepended ot the post text

        Args: 
            do_load_spacy (bool, optional): whether or not we load the spacy library, defaults to True.
    """

    if CS.USE_MINIFIED_DATA:
        lg.warning(
            "Using minified data (fraction: {0})".format(CS.MINIFY_FRAC))

    if not hasattr(globals(), 'reddit') and not single_post:
        load_reddit_settings()

    if not hasattr(globals(), 'df_posts') and not single_post:
        if CS.LOAD_POSTS:
            load_posts()
        else:
            lg.warn("Skipping loading posts...")

    if not hasattr(globals(), 'df_comments') and not single_post:
        if CS.LOAD_COMMENTS:
            load_comments()
        else:
            lg.warning("Skipping loading comments...")

    if not hasattr(globals(), 'df_liwc') and not hasattr(globals(), 'df_liwc_merged') and not single_post:
        if CS.LOAD_LIWC:
            if CS.TITLE_AS_STANDALONE:
                load_liwc()
                load_liwc_title()
            else:
                load_liwc_merged()
        else:
            lg.warning("Skipping loading liwc...")

    if not hasattr(globals(), 'df_foundations') and not hasattr(globals(), 'df_foundations_merged') and not single_post:
        if CS.LOAD_FOUNDATIONS:
            if CS.TITLE_AS_STANDALONE:
                load_foundations()
                load_foundations_title()
            else:
                load_foundations_merged()
        else:
            lg.warning("Skipping loading foundations...")

    if not hasattr(globals(), 'nlp'):
        # and len([item for sublist in CS.FEATURES_TO_GENERATE_MONO["writing_sty"] for item in sublist]) > 0 or CS.DO_TOPIC_MODELLING:
        if do_load_spacy:
            load_spacy()

    if CS.TITLE_AS_STANDALONE and not single_post:
        add_standalone_title_features()


def load_reddit_settings():
    """Load the reddit settings from secrets/reddit.json
    """
    global reddit
    path = sys.path[0]+"/secrets/reddit.json"
    if exists(path):
        file = open(path)
        data = json.load(file)
        id = data[CS.REDDIT_INSTANCE_IDX]["client_id"]
        secret = data[CS.REDDIT_INSTANCE_IDX]["client_secret"]
        agent = data[CS.REDDIT_INSTANCE_IDX]["user_agent"]
        reddit = praw.Reddit(
            client_id=id,
            client_secret=secret,
            user_agent=agent
        )
    else:
        print("Reddit secret not found. Skipping")


def load_comments():
    """Load the comments csv
    """
    global df_comments
    lg.info("Loading comments: "+CS.COMMENTS_CLEAN)
    df_comments = pd.read_csv(CS.COMMENTS_CLEAN, index_col=False)


def load_posts():
    """Load the posts csv
    """
    global df_posts
    lg.info("Loading posts: "+CS.POSTS_CLEAN)

    # Done 0-110000,
    df_posts = pd.read_csv(CS.POSTS_CLEAN, index_col=False)
    #df_posts = pd.read_csv(CS.POSTS_CLEAN, index_col=False, nrows=110000)
    #df_checked = pd.read_csv("/mnt/c/Users/Philipp/Desktop/ids.csv")
    #df_posts = df_posts[~df_posts['post_id'].isin(df_checked["post_id"].tolist())]

    lg.info(f"{len(df_posts)} number of posts")
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
    df_foundations_title = df_foundations_title.add_prefix(
        CS.FOUNDATIONS_TITLE_PREFIX)


def load_foundations_merged():
    """Load the moral foundations csv for the posts with title preprended
    """
    global df_foundations_merged
    lg.info("Loading foundations: "+CS.FOUNDATIONS_MERGED)
    df_foundations_merged = pd.read_csv(CS.FOUNDATIONS_MERGED, index_col=False)
    df_foundations_merged = df_foundations_merged.add_prefix(
        CS.FOUNDATIONS_MERGED_PREFIX)


def load_liwc():
    """ Load the liwc csv for the posts only
    """
    global df_liwc
    lg.info("Loading liwc posts: "+CS.LIWC)
    df_liwc = pd.read_csv(CS.LIWC, index_col=False)
    df_liwc = df_liwc.add_prefix(CS.LIWC_PREFIX)


def load_liwc_title():
    """Load the liwc csv for the titles only
    """
    global df_liwc_title
    lg.info("Loading liwc titles: "+CS.LIWC_TITLE)
    df_liwc_title = pd.read_csv(CS.LIWC_TITLE, index_col=False)
    df_liwc_title = df_liwc_title.add_prefix(CS.LIWC_TITLE_PREFIX)


def load_liwc_merged():
    """ Load the liwc csv for the posts with title preprended
    """
    global df_liwc_merged
    lg.info("Loading liwc merged: "+CS.LIWC_MERGED)
    df_liwc_merged = pd.read_csv(CS.LIWC_MERGED, index_col=False)
    df_liwc_merged = df_liwc_merged.add_prefix(CS.LIWC_MERGED_PREFIX)


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
