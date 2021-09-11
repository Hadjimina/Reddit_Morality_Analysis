
import pandas as pd
import logging as lg
import coloredlogs
coloredlogs.install()

def enforce_requirements(df):
    """ Enforce some requirments on our df. Usually done before plotting (NOT BEFORE CALCULATING THE FEATURES!)

    returns : df
        df: Dataframe with enforce requirements (probably smaller than the original dataframe)

    Parameters
    ----------
    df : dataframe
        dataframe that contains all the features in its columns

    """


    old_df_size = df.shape[0]
    cols = list(df.columns)

    # Requirement 1: Post must have at least 3 votes
    abs_vote_cols = [i for i in cols if "reactions" in i and i.split("_")[1].isupper()]
    if len(abs_vote_cols) > 5:
        raise Exception("Enforece requirement 1: too many values in abs_vote_cols ({0})".format(str(abs_vote_cols)))
    
    if len(abs_vote_cols) > 0:
        df = df[sum([df[i] for i in abs_vote_cols]) > 3]
    else:
        lg.warning("Minium number of votes not enforces")

    # Requirement 2: Post must have 3 upvotes at least
    # TODO

    new_df_size = df.shape[0]
    new_size = round(new_df_size/old_df_size,3)*100
    if new_size < 1:
        lg.warning("By enforcing requirments df reduced to {0}% (new size = {1})".format(new_size, new_df_size))
    return df