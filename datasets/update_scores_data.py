import sys 
sys.path.append('..')
import logging as lg
import re
import numpy as np
import coloredlogs
import constants as CS
import pandas as pd
from helpers import *
from tqdm import tqdm
import helpers.globals_loader as globals_loader
coloredlogs.install()

def update_score_elem(do_posts=True, overwrite=True):
    """ Iterate over all post ids and append the upvotes and downvotes to the dataframe

    Args:
        do_posts: Whether we should get the new data for posts or not (False will get the updated comment scores)
        overwrite: Whether the original dataframe csv should be overwritten
    """
    prefix = "post" if do_posts else "comment"
    df = globals_loader.df_posts if do_posts else globals_loader.df_comments
    ids = df[prefix+"_id"].tolist()

    reddit = globals_loader.reddit
    elem_prefix = "t3" if do_posts else "t1"
    ids = [i if i.startswith(elem_prefix+"_") else f"{elem_prefix}_{i}" for i in ids]
    scores = []
    ratios = []

    
    if do_posts:
        for i in tqdm(range(len(ids))): #in tqdm(reddit.info(ids), total=len(ids)):
            submission = reddit.info(ids[i])
            #TODO
            scores.append(submission.score)
            ratios.append(submission.upvote_ratio)
            # see https://api.reddit.com/api/info/?id=t3_apcnyn
    else:
        for i in tqdm(range(len(ids)))
        #for comment in tqdm(reddit.info(ids), total=len(ids)):
            comment = reddit.info(ids[i])   
            scores.append(comment.score)
            # comments do not have an upvote ratio
            # see https://api.reddit.com/api/info/?id=t1_eg7dfxx

    if do_posts:
        ups_downs = np.array(list(map(lambda s, r: helper_functions.get_ups_downs_from_ratio_score(r, s) , scores, ratios)))
        df[prefix+"_ups"] = ups_downs[:,0].tolist()
        df[prefix+"_downs"] = ups_downs[:,1].tolist()
    else:
        df["comment_score"] = scores
        
    save_dir = CS.POSTS_CLEAN if do_posts else CS.COMMENTS_RAW
    if not overwrite:
        save_dir = save_dir.replace(".", "_updated.")
    lg.info(f"Saving updated {prefix}s to {save_dir}")
    df.to_csv(save_dir, index=False)

def update_comments():
    """ Updates the score column on the comments dataframe new data from reddit.
        Calls are made in batches of 100.
    """
    lg.info("Updating scores on comments")
    CS.LOAD_POSTS = False
    CS.LOAD_COMMENTS = True
    CS.LOAD_FOUNDATIONS = False
    CS.LOAD_LIWC = False
    globals_loader.init()

    update_score_elem(do_posts=False)

def update_posts():
    """ Add post_ups and post_downs column to the posts dataframe by getting the new data from reddit.
        Calls are made in batches of 100.
    """
    lg.info("Updating up and down votes on posts")
    CS.LOAD_POSTS = True
    CS.LOAD_COMMENTS = False
    CS.LOAD_FOUNDATIONS = False
    CS.LOAD_LIWC = False
    globals_loader.init()

    update_score_elem()

if __name__ == "__main__":
    
    if "-comments" in sys.argv or "-com" in sys.argv:
        update_comments()
    elif "-posts" in sys.argv:
        update_posts()
    else:
        update_comments()
        update_posts()
        
    

    