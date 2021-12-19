
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
    fn_executed = [] # Why is this needed at all?
    sent_msgs =[]
    for category in feat_to_gen:
        # We iterate over every column in dataset s.t. we first use all columns that use e.g. "post_author", before moving on
        for i in range(len(sub_df.columns)):
            for feature_tuple in feat_to_gen[category]:
                
                funct = feature_tuple[0]
                col_name = feature_tuple[1]
                fn_name = funct.__name__
                idx = list(sub_df).index(col_name)
                
                if idx == i:
                    #col = sub_df.iloc[:,idx] #TODO: change this to use column text (e.g. "post_text") instead of index
                    col = sub_df[col_name]
                    if not fn_name in fn_executed:
                        msg = "Started {fn} on {host}".format(fn = funct.__name__, host=socket.gethostname())
                        if CS.NOTIFY_TELEGRAM and (id == 0 or str(id) == "mono") and msg not in sent_msgs:
                            sent_telegram_notification(msg)
                            sent_msgs.append(msg)
                     
                        feature_df_list.append(feature_to_df(id, category, col, funct))
                        
                        if CS.TITLE_AS_STANDALONE and col == CS.POST_TEXT:
                            feature_df_list.append(feature_to_df(id, "title_"+category, sub_df[CS.POST_TITLE], funct))
                            
                        fn_executed.append(fn_name)
                        #msg = "    Finished {fn} on {host}".format(fn = funct.__name__, host=socket.gethostname())
                        #if CS.NOTIFY_TELEGRAM and (id == 0 or str(id) == "mono") and msg not in sent_msgs:
                        #    sent_telegram_notification(msg)
                        #    sent_msgs.append(msg)
    
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
    if id == 0:
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