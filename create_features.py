# styleguide:
## https://www.python.org/dev/peps/pep-0008/
## https://realpython.com/documenting-python-code/
#Logging
## https://realpython.com/python-logging/

import logging as lg
from datetime import datetime

import coloredlogs
import pandas as pd
import praw
from tqdm import tqdm

import constants as CS
import globals_loader
import helpers.df_visualisation as vis
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *

coloredlogs.install()

def generate_report(df):
    """Generate report by visualising the columns of the dataframe as 
        histograms and listing 3 example post (low, medium, high value)
        below the histograms. Saves as report.png in home directory of script

    Parameters
    ----------
    df : Dataframe
        Dataframe with format of feature_to_df

    """
    lg.info("Generating report")

    vis.df_to_plots(df)
    vis.df_to_text_png(df)

def feature_to_df(category, column, funct):
    """Generate dataframe out of return value of category name, column data and feature function

    returns : dataframe
        dataframe with headers corresponding to category and feature function 
        e.g "speaker" + "author_age" = "speaker_author_age",
        values are int returned from feature function

    Parameters
    ----------
    category : str
        Which feature category we are currently using

    column: [str]
        dataframe column we apply the feature function to

    funct: str->[(str, count)]
        feature function

    """
    lg.info('Running "{0}"'.format(funct.__name__))

    temp_s = column.progress_apply(funct) # [(a,#)....]
    #temp_s = column.parallel_apply(funct)
    fst_value = temp_s.iat[0]
    cols = ["{0}_{1}".format(category,tpl[0]) for tpl in fst_value]
    temp_s = temp_s.apply(lambda x: [v for s,v in x])
    df = pd.DataFrame(temp_s.to_list(), columns=cols)
    return df

def main():
    """Iterate over all posts and create features dataframe"""
    
    tqdm.pandas()

    #initalize reddit object for PRAW
    global reddit
    reddit = praw.Reddit(
        client_id="ChMem9TZYJif1A",
        client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
        user_agent="android:com.example.myredditapp:v1.2.3"
    )

    features_to_generate = {
        "speaker":[
            #(get_author_amita_post_activity, CS.POST_AUTHOR),
            #(get_author_age, CS.POST_AUTHOR),
            #(get_post_author_karma, CS.POST_AUTHOR)
        ], 
        "writing_sty":[
            (get_punctuation_count, CS.POST_TEXT)
        ],
        "behaviour":[
            
        ],
        "reactions":[
            (get_judgement_labels, CS.POST_ID)
        ]
    }
    

    lg.info("Loading Clean posts: "+CS.POSTS_CLEAN)
    df_post_cleaned = pd.read_csv(CS.POSTS_CLEAN, index_col=False)

    # Create a list of all individual feature dfs and merge. Lastly append last column with post_id
    feature_df_list = []
    for category in features_to_generate:
        # We iterate over every column in dataset s.t. we first use all columns that use e.g. "post_author", before moving on
        for i in range(len(df_post_cleaned.columns)):
            for feature_tuple in features_to_generate[category]:
                funct = feature_tuple[0]
                idx = feature_tuple[1]
                if idx == i:
                    col = df_post_cleaned.iloc[:,idx]
                    feature_df_list.append(feature_to_df(category, col, funct))

    feature_df_list.append(df_post_cleaned["post_id"])
    feature_df_list.append(df_post_cleaned["post_text"])

    feature_df = pd.concat(feature_df_list, axis=1,join="inner")       
    
    # Create histogram and sample texts as png
    vis.df_to_plots(feature_df)

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y")
    feature_df = feature_df.drop("post_text",axis=1)
    feature_df.to_csv(CS.OUTPUT_DIR+"features_output_"+date_time+".csv")
    
    
if __name__ == "__main__":
    globals_loader.init()  
    main()
