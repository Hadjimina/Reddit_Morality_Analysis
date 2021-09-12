"""
helper_functions.py
"""
import helpers.globals_loader as globals_loader

def dict_to_feature_tuples(dict, suffix=""):
    """Take a dict at end of a feature function and convert it to a tuple format for the dataframe

    Args:
        dict (dictionary): feature dictionary

    Returns:
        [(feature name, value)]: [description]
    """
    return [(("{0}"+suffix).format(k), v) for k, v in dict.items()]

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


def get_abs_and_perc_dict(abs_dict, out_off_ratio, append_abs=True):
    """Get a feature dictinoary containing only absolute values and extended it to include the percentage values aswell.

    Args:
        abs_dict ({str: int}): dict of calucluated features with only absolute features

    Returns:
        {str:int}: dict of calucluated features with absolute and percentage features
    """ 

    features = list(abs_dict.keys())
    #create abs and perc values
    all_keys = [x+"_abs" for x in features] + [x+"_perc" for x in features] 
    complete_dict = dict.fromkeys(all_keys,0)

    #nr_sents = len(doc.sents)
    for v in features:
        curr_value = abs_dict[v]
        abs_postpend = "_abs" if append_abs else ""
        complete_dict[v+abs_postpend] = curr_value
        complete_dict[v+"_perc"] = curr_value/out_off_ratio
    return complete_dict