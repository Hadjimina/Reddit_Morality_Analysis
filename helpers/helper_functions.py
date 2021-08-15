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
