import logging as lg
import re
import coloredlogs
coloredlogs.install()

# import global vars

def get_punctuation_count(post_text):
    """Count how many times certain punctuations syombols occur in text

    returns quotation_count, question_count, exclamation_count

    Parameters
    ----------
    post_text : str
        Full body text of r/AITA post

    """

    
    symbol_dict = {"!":0,'"':0,"?":0}
    
    for i in range(len(symbol_dict.keys())):
        symbol = list(symbol_dict.keys())[0]
        print(symbol)
        regex = re.compile("[{0}]".format(symbol))
        print(regex)
        count = len(regex.search(post_text).group(0))
        symbol_dict[symbol] = count

    tuple_list = [("{0}_count".format(k), v) for k, v in symbol_dict.items()]


    return amita_activity_counter

