"""
helper_functions.py
"""

def dict_to_feature_tuples(dict, suffix=""):
    """Take a dict at end of a feature function and convert it to a tuple format for the dataframe

    Args:
        dict (dictionary): feature dictionary

    Returns:
        [(feature name, value)]: [description]
    """
    return [(("{0}"+suffix).format(k), v) for k, v in dict.items()]