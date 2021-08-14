from helpers.helper_functions import dict_to_feature_tuples
import logging as lg
import re

import coloredlogs
import constants as CS
import pandas as pd
from helpers import *
import helpers.globals_loader as globals_loader
coloredlogs.install()

def string_to_lower_alphanum(str):
    """Convert any string to lowercase and only containing alphanumeric characters + whitespaces

    Args:
        str (string): string to convert
    Returns:
        str (string): lowercase string with only alphanumeric characters + whitespaces
    """    

    # to lower case
    str = str.lower()
    # remove non alphanumeric characters (and whitespaces)
    str = re.sub(r'[^A-Za-z0-9\s]+', '', str)
    return str


def count_label(comment, acronym):
    """Check if the judgemnt label (YTA, NTA, INFO, ESH, NAH) is contained in comment.
       We not only check for exact label match but also some minor text analysis

    Args:
        comment (string): post comment body
        label (string): label to detect
    Returns:
        middle_dist (int): distance of label detection from middle of comment
    """    

    # split on any whitspace
    word_list = comment.split()
    label_i = CS.JUDGMENT_ACRONYM.index(acronym)

    # naive detection
    

    # priorities values that are in the beginning or end
    middle_index = len(comment)//2
    middle_dist = 0
    try:
        middle_dist = abs(middle_index - word_list.index(acronym))

    except ValueError as e:
        middle_dist = 0
        # specific label not found
        #TODO: Some more sophisticated checking needed to check expressions (i.e. "you are the asshole")


    return middle_dist
    

def get_judgement_labels(post_id):
    """Returns judgemnt label counts (YTA, NTA, INFO, ESH, NAH)

    Args:
        post_id (int): Id of the reddit post

    Returns:
        [(str, int)]: e.g. [("NTA",10), ("YTA", 20),...]
    """
    #if not hasattr(globals_loader, 'df_comments'):
    #    globals_loader.load_comments()
    
    df_comments = globals_loader.df_comments
    df_comments = df_comments.loc[df_comments["post_id"] == post_id]
    df_comments = df_comments[["comment_text", "comment_score"]]
    
    # Create accumulation dict
    label_counter = CS.JUDGMENT_ACRONYM + ["weighted_"+s for s in CS.JUDGMENT_ACRONYM]
    label_counter = dict.fromkeys(label_counter,0)
    
    
    for i, comment_row in enumerate(df_comments.itertuples(), 1):
        _, comment_body, score = comment_row
        comment_body = string_to_lower_alphanum(str(comment_body))

        # Check votes for each different label in one single comment
        # Maybe somebody "votes twice" within one comment
        cur_label_dist = {}
        for i in range(len(CS.JUDGMENT_ACRONYM)):
            acronym = CS.JUDGMENT_ACRONYM[i]
            middle_dist = count_label(comment_body, acronym)
            cur_label_dist[acronym] = middle_dist
        
        # Only count vote of label furthest away from middle
        #  (Sort dict by highest value)
        cur_label_dist = dict(sorted(cur_label_dist.items(), key=lambda item: item[1]))

        vote = list(cur_label_dist.values())[0]
        if vote > 0:
            weighted_vote = vote * score
            label_counter[acronym.upper()] += vote
            label_counter["weighted_"+acronym.upper()] += weighted_vote
    
    tuple_list =  dict_to_feature_tuples(label_counter)

    return tuple_list

