from helpers.helper_functions import dict_to_feature_tuples
import logging as lg
import re

import coloredlogs
import constants as CS
import pandas as pd
from helpers import *
from helpers.helper_functions import *
import helpers.globals_loader as globals_loader
coloredlogs.install()

def check_crossposts(post_id):
    """ Check if post has been crossposted to r/AmITheDevil or r/AmITheAngel, signifiying obvious wrong doing or correct behaviour respectively 

    Args:
        post_id (string): id of post

    Returns:
        feature_list: list of features tuples e.g. [("account_age", age), ("account_comment_karma", comment_karma)]
    """ 
    reddit = globals_loader.reddit
    post = reddit.submission(post_id)
    sub_list = []
    in_devil = False 
    in_angel = False
    for duplicate in post.duplicates():
        sub_list.append(duplicate.subreddit.display_name.lower())

    is_devil = int("amithedevil" in sub_list)
    is_angel = int("amitheangel" in sub_list)

    feature_list = [("is_angel", is_angel), ("is_devil", is_devil)]
    return feature_list

def count_label(comment, acronym):
    """Check if the judgemnt label (YTA, NTA, INFO, ESH, NAH) is contained in comment.
       We not only check for exact label match but also some minor text analysis

    Args:
        comment (string): post comment body
        label (string): label to detect
    Returns:
        middle_dist (int): distance of label detection from middle of comment in percent 
            0 = detected acryonm in center of post text
            1 = detected acryonm either at beginning or end of comment
    """    

    # split on any whitspace
    label_i = CS.JUDGMENT_ACRONYM.index(acronym)
    # naive detection
    

    # priorities values that are in the beginning or end
    middle_index = len(comment)//2
    middle_dist = 0
    try:
        dist_abs = abs(middle_index - comment.index(acronym.lower()))
        middle_dist = dist_abs/(len(comment)/2)
        
    except ValueError as e:
        middle_dist = -1
        # specific label not naively found
        #TODO: Some more sophisticated checking needed to check expressions (i.e. "you are the asshole")


    return middle_dist
    
def get_judgement_labels(post_id):
    """Returns judgement label counts (YTA, NTA, INFO, ESH, NAH)

    Args:
        post_id (int): Id of the reddit post

    Returns:
        tuple list (list): e.g. [("NTA",10), ("YTA", 20),...]
    """

    df_comments = globals_loader.df_comments
    df_comments = df_comments.loc[df_comments["post_id"] == post_id]
    df_comments = df_comments[["comment_text", "comment_score"]]

    label_counter = CS.JUDGMENT_ACRONYM + ["weighted_"+s for s in CS.JUDGMENT_ACRONYM]
    label_counter = dict.fromkeys(label_counter,0)
    
    
    for i, comment_row in enumerate(df_comments.itertuples(), 1):
        _, comment_body, score = comment_row
    
        comment_body = get_clean_text(str(comment_body), None, do_lemmatization=False)
        comment_body_no_punct = get_clean_text(str(comment_body), None, remove_punctuation=2, do_lemmatization=False)

        if any(list(map(lambda x: x.lower() in comment_body, CS.BOT_STRINGS))):
            #print("___SKIPPED___")
            continue


        #print("---start---")
        #print(comment_body)
        labels_loc = {}

        middle = max(len(comment_body)//2,1)
        middle_simple = max(len(comment_body_no_punct.split())//2,1)
        for k in CS.JUDGEJMENT_DICT.keys():
            idxes = []              # "e.g. You are the asshole"
            center_dist = []                 
            idxes_simple = []       # e.g. "YTA"
            center_dist_simple = []   
            for x in string_matching_arr_append_ah(CS.JUDGEJMENT_DICT[k]):   
                if len(x.split()) > 1: 
                    idxes = find_all(comment_body, x.lower())
                    idxes = list(filter(lambda x: x != -1, idxes))
                    # No longer index but distance from center
                    center_dist = list(map(lambda q: (abs(middle-q) / middle), idxes ))
                    center_dist.sort(reverse=True)
                else: 
                    idxes_simple = [i for i,y in enumerate(comment_body_no_punct.split()) if y==x.lower()]
                    # No longer index but distance from center
                    center_dist_simple = list(map(lambda q: (abs(middle_simple-q) / middle_simple), idxes_simple ))
                    center_dist_simple.sort(reverse=True)
                
            
            # Order by distance
            merged = center_dist + center_dist_simple
            merged.sort(reverse=True)
            labels_loc[k] = merged

        # Check if more than one vote was detected
        nr_votes = len(flatten_list(list(labels_loc.values())))
        vote = ""
        if nr_votes > 1:
            # We remove info since this could often cause errors and we are not super interested in it
            labels_loc.pop("INFO", None)
            max_label = ""
            max_value = 0
            for k in labels_loc.keys():
                if len(labels_loc[k]) > 0 and labels_loc[k][0] > max_value:
                    max_value = labels_loc[k][0]
                    max_label = k
        
            vote = max_label
        else:
            # Take first dict entry that contains some value
            for k in labels_loc.keys():
                if len(labels_loc[k]) > 0:
                    vote = k
    
        #print(labels_loc)
        #print(vote)
        #print("___end___")
        if vote != "":
            label_counter[vote.upper()] += 1
            label_counter["weighted_"+vote.upper()] += int(score)
    
    tuple_list =  dict_to_feature_tuples(label_counter) 

    return tuple_list



