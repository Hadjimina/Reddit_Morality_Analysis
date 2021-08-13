from helpers.helper_functions import dict_to_feature_tuples
import logging as lg
import re
import coloredlogs

from helpers import *
coloredlogs.install()

# import global vars

def get_punctuation_count(post_text):
    """Count how many times certain punctuations syombols occur in text

    Args:
        post_text ([str]): Full body text of r/AITA post

    Returns:
        [(str, int)]:  e.g. ("!_count": 10)
    """    

    symbols = ["!",'"', "?"]
    symbol_dict = dict.fromkeys(symbols, 0)

    #filter out hyperlinks
    post_text = re.sub(r'http\S+', '', post_text)
    
    for i in range(len(symbol_dict.keys())):
        
        symbol = list(symbol_dict.keys())[i]
        symbol_dict[symbol] = post_text.count(symbol)

    tuple_list = dict_to_feature_tuples(symbol_dict, "_count")
    #tuple_list = [("{0}_count".format(k), v) for k, v in symbol_dict.items()]
    return tuple_list

