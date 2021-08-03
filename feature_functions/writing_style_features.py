import logging as lg

import coloredlogs

coloredlogs.install()

# import global vars

def get_punctuation_count(post_text):
    """Count how many times certain punctuations syombols occur in text

    returns : [(str, int), ...]
        e.g. ("!_count": 10)

    Parameters
    ----------
    post_text : str
        Full body text of r/AITA post

    """
    symbols = ["!",'"', "?"]
    symbol_dict = dict.fromkeys(symbols, 1)
    print(post_text)
    for i in range(len(symbol_dict.keys())):
        
        symbol = list(symbol_dict.keys())[i]
        symbol_dict[symbol] = post_text.count(symbol)

    tuple_list = [("{0}_count".format(k), v) for k, v in symbol_dict.items()]

    return tuple_list

