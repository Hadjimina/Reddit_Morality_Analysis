"""
helper_functions.py
"""
import globals_loader 

def dict_to_feature_tuples(dict, suffix=""):
    """Take a dict at end of a feature function and convert it to a tuple format for the dataframe

    Args:
        dict (dictionary): feature dictionary

    Returns:
        [(feature name, value)]: [description]
    """
    return [(("{0}"+suffix).format(k), v) for k, v in dict.items()]



def setup_load_dfs():
    if not hasattr(globals_loader, 'df_comments'):
        globals_loader.load_comments()

    if not hasattr(globals_loader, 'df_posts'):
        globals_loader.load_posts()