"""
helper_functions.py
"""
import helpers.globals_loader as globals_loader
import numpy as np
import re
from datetime import datetime


def get_ups_downs_from_ratio_score(r,s):
    """ Given a ratio and a score, we will return individual number of upvotes and downvotes
        We have to do this since, reddit remove the possibility to check the exact amount of upvotes and downvotes a few years ago. 
        Also the ratio is rounded so we will not get completely accurate values.

        We get formula since we know that x-y=s and x/x+y = r
    Args:
        ratio: upvote ratio from reddit
        score: the current score of the post/comment

    Returns:
        ups, downs: ups = number of upvotes, downs = number of downvotes
    """

    ups = round((s - r * s) / ( 2 * r - 1 ) + s) if r != 0.5 else 0 #if we have have a 50% upvote ratio we set the upvotes to 0
    downs = round(ups - s)

    return [ups, downs]

def dict_to_feature_tuples(dict, suffix=""):
    """ Take a dict at end of a feature function and converts it into the tuple format suitable for the dataframe

    Args:
        dict: dictionary containing features data
        suffix: string we post pend to each feature category

    Returns:
        tuple_list: list of tuples for the dataframe e.g. [(feature name, value),...]
    """
    tuple_list = []
    #all_values_zero = True
    for k,v in dict.items():
        tpl = (("{0}"+suffix).format(k), v)
        #if not np.isclose(v, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        #    all_values_zero = False
        tuple_list.append(tpl)

    return tuple_list #if not all_values_zero else []

def prep_text_for_string_matching(text):
    """Prepare text for string matching in two steps
        1. make all text lowercase
        2. replace multiple whitespace with only one whitespace
    Args:
        text (str): some text we want to perform string matching on

    Returns:
        str: text prepared for string matching
    """    
    text = text.lower()
    
    text = " ".join(text.split())
    return text

def string_matching_arr_append_ah(matching_list):
    """When performing string matching we sometimes manualy specific which words to match. Often these include the words "asshole".
        Often users on AITA, do not write "asshole" but write "ah" instead. Thus we need to extend the matching list but replace all "asshole" occurences with "ah"

    Args:
        matching_list ([str]): list of matching strings, these do not include any "ah" yet but only "asshole"

    Returns:
        [str]: list of matching strings extended to include "asshole" and "ah
    """    

    asshole_str = "asshole"
    ah_str = "ah"

    ah_to_post_pend = []
    for match_str in matching_list:
        if asshole_str in match_str:
            ah_to_post_pend += match_str.replace(asshole_str, ah_str)

    return matching_list


def get_abs_and_norm_dict(abs_dict, out_off_ratio, append_abs=True, only_norm=False):
    """Get a feature dictinoary containing only absolute values and extended it to include the normalised values aswell.

    Args:
        abs_dict ({str: int}): dict of calucluated features with only absolute features
        append_abs (bool): whether or not we should append the string "_abs" to the exisitng keys in abs_dict
        only_norm (bool): whether or not we should return only the normalised values

    Returns:
        {str:int}: dict of calucluated features with absolute and normalised features
    """ 
    features = list(abs_dict.keys())
    #create abs and perc values
    abs_postpend = "_abs" if append_abs else ""
    all_keys = [x+"_norm" for x in features] 
    if not only_norm:
        all_keys = [x+abs_postpend for x in features] + all_keys
    complete_dict = dict.fromkeys(all_keys,0)

    for v in features:
        curr_value = abs_dict[v]
        if not only_norm:
            complete_dict[v+abs_postpend] = curr_value
        complete_dict[v+"_norm"] = curr_value/max(out_off_ratio,1)
    return complete_dict

def get_date_str():
    """ Get current date in format dd_mm_YYYY

    Returns:
        date_time: date string
    """
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y")
    return date_time