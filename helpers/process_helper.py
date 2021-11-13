
import pandas as pd
import constants as CS
import logging as lg
from tqdm import tqdm
import socket
from helpers.helper_functions import *
tqdm.pandas()

def process_run(feat_to_gen, sub_df, id):
    """ Apply all functions of "features_to_generate" to each row of subsection of dataframe

    Args:
        feat_to_gen: list of which features should be generated. List consist of tuple, first being the feature function and the second being the argument said function requires
        sub_df: Sub section of all posts. They are split up to allow for parallel processing.
        id: Thread id

    Returns:
        feature_df_list: list of dataframes that contain the extracted features
    """  

    # Create a list of all individual feature dfs and merge. Lastly append last column with post_id
    feature_df_list = []
    for category in feat_to_gen:
        # We iterate over every column in dataset s.t. we first use all columns that use e.g. "post_author", before moving on
        for i in range(len(sub_df.columns)):
            for feature_tuple in feat_to_gen[category]:
                
                funct = feature_tuple[0]
                idx = feature_tuple[1]
                
                msg = "Started {fn} on {host}".format(fn = funct.__name__, host=socket.gethostname())
                if CS.NOTIFY_TELEGRAM and id == 0:
                    sent_telegram_notification(msg)
                if idx == i:
                    col = sub_df.iloc[:,idx] #TODO: change this to use column text (e.g. "post_text") instead of index
                    tmp_cat = "title_"+category if CS.TITLE_AS_STANDALONE and idx == CS.POST_TITLE else category
                    feature_df_list.append(feature_to_df(id, tmp_cat, col, funct)) 
                    #tmp_df = pd.concat(feature_df_list, axis=1)                
                    #tmp_df.index = sub_df.index
                    #tmp_df.to_csv(CS.TMP_SAVE_DIR+"/thread_{0}_tmp.csv".format(id))

                # Make standalone title feature for each feature on post text
                #if CS.TITLE_AS_STANDALONE and idx == CS.POST_TEXT:
                #    col = sub_df.iloc[:,CS.POST_TITLE] #TODO: change this to use column text (e.g. "post_text") instead of index
                #    feature_df_list.append(feature_to_df(id, "title_"+category, col, funct))
                #    #tmp_df = pd.concat(feature_df_list, axis=1)                
                #    #tmp_df.index = sub_df.index
                #    #tmp_df.to_csv(CS.TMP_SAVE_DIR+"/thread_{0}_tmp.csv".format(id))

    
    # Post pends some post specific information
    if id == CS.MONO_ID:
        post_pend = sub_df[CS.POST_PEND_MONO]
    else:
        post_pend = sub_df[CS.POST_PEND]
    #TODO: here we assume that order is kept and never switched up => also pass ids into process_run and merge based on id in the end to be sure
    post_pend.reset_index(drop=True, inplace=True)
    feature_df_list.append(post_pend)
    return feature_df_list


def feature_to_df(id, category, column, funct):
    """Generate dataframe out of return value of category name, column data and feature function

    Args:
        category (string): Which feature category we are currently using
        column  (string): dataframe column we apply the feature function to
        function (any->any): feature function
    Returns: 
        dataframe: dataframe with headers corresponding to category and feature function. e.g "speaker" + "author_age" = "speaker_author_age", values are int returned from feature function
    """
    lg.info('Running "{0}" on thread {1}'.format(funct.__name__, id))
    if id == "mono":
        spacy_fn_names = [fn.__name__ for fn in CS.SPACY_FUNCTIONS]
        lg.info ("  Spacy features: {0}".format(spacy_fn_names))

    temp_s = column.progress_apply(funct)
    fst_value = temp_s.iat[0]
    cols = ["{0}_{1}".format(category, tpl[0]) for tpl in fst_value]
    temp_s = temp_s.apply(lambda x: [v for s,v in x])
    df = pd.DataFrame(temp_s.to_list(), columns=cols)
    
    return df