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

def count_label(comment, acronym, word_list):
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
        dist_abs = abs(middle_index - word_list.index(acronym.lower()))
        middle_dist = dist_abs/(len(comment)/2)
        
    except ValueError as e:
        middle_dist = -1
        # specific label not naively found
        #TODO: Some more sophisticated checking needed to check expressions (i.e. "you are the asshole")


    return middle_dist
    
def get_judgement_labels(post_id):
    """Returns judgemnt label counts (YTA, NTA, INFO, ESH, NAH)

    Args:
        post_id (int): Id of the reddit post

    Returns:
        [(str, int)]: e.g. [("NTA",10), ("YTA", 20),...]
    """

    
    df_comments = globals_loader.df_comments
    df_comments = df_comments.loc[df_comments["post_id"] == post_id]
    df_comments = df_comments[["comment_text", "comment_score"]]
    #print(post_id)
    #if  df_comments.empty:
    #    print('DataFrame empty!')
    # Create accumulation dict
    label_counter = CS.JUDGMENT_ACRONYM + ["weighted_"+s for s in CS.JUDGMENT_ACRONYM]
    label_counter = dict.fromkeys(label_counter,0)
    
    
    for i, comment_row in enumerate(df_comments.itertuples(), 1):
        _, comment_body, score = comment_row
        comment_body = string_to_lower_alphanum(str(comment_body))
        #print(comment_body)
        # Check votes for each different label in one single comment
        # Maybe somebody "votes twice" within one comment
        cur_label_dist = {}
        word_list = comment_body.split()
        for i in range(len(CS.JUDGMENT_ACRONYM)):
            acronym = CS.JUDGMENT_ACRONYM[i]
            middle_dist = count_label(comment_body, acronym, word_list)
            cur_label_dist[acronym] = middle_dist
        #print(cur_label_dist)
        # Only count vote of label furthest away from middle
        #  (Sort dict by highest value)
        cur_label_dist = dict(sorted(cur_label_dist.items(), key=lambda item: item[1], reverse=True))
        
        vote = list(cur_label_dist.keys())[0]
        dist = list(cur_label_dist.values())[0]

        if dist >= 0:
            label_counter[vote.upper()] += 1
            label_counter["weighted_"+vote.upper()] += int(score)
    
    ''' Calculate controversy score as a ratio of postive judgemnte to overall judgement count.
         We want to see how big the diffeence in opinion is.
         Score of 1 = 0.5, 0.5 vote ratio split => big controversy
         Score of 0 = 0 or 1 pos_score ratio => no controversy
    '''
    # TODO: controversy score
    """ pos_score = label_counter["NAH"]+label_counter["NTA"]
    neg_score = label_counter["YTA"]+label_counter["ESH"]
    pos_score_weighted = label_counter["weighted_NAH"]+label_counter["weighted_NTA"]
    neg_score_weighted = label_counter["weighted_YTA"]+label_counter["weighted_ESH"]
    if pos_score+neg_score >0:
        label_counter["controversy"] = 1-2*abs(pos_score/(pos_score+neg_score)-0.5)
    
    if pos_score_weighted+neg_score_weighted > 0:
        label_counter["weighted_controversy"] = 1-2*abs(pos_score_weighted/(pos_score_weighted+neg_score_weighted)-0.5)
    """    
    #print(label_counter)
    tuple_list =  dict_to_feature_tuples(label_counter) 

    return tuple_list



