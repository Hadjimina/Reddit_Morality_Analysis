# styleguide:
# https://www.python.org/dev/peps/pep-0008/
# https://realpython.com/documenting-python-code/
# Logging
# https://realpython.com/python-logging/

import logging as lg
import queue
import sys
from multiprocessing import Queue, set_start_method
from os import walk
from pathlib import Path

import coloredlogs
import numpy as np
import pandas as pd
from better_profanity import profanity
import socket
import json
import subprocess
import shutil

import constants as CS
import distributed_configuration as dist_conf
import helpers.df_visualisation as vis
import helpers.globals_loader as globals_loader
import parallel_process as p_process
from feature_functions.topic_modelling import *
from feature_functions.reaction_features import *

from helpers.helper_functions import *
from helpers.process_helper import *

coloredlogs.install()


def set_features_to_run_dist(title_as_standalone):
    """Overwrite feature functions and similar params in constants (i know...overwriting "constants") 
       according to dictionary in distributed_configuration.py. 
       This should only be used when extracting features in a distributed fashion i.e. 2+ pc at the same time

        Args:
            titile_as_standalone (bool): Whether or not we should treat the post title as standlone or prepend it to all post_texts
    """

    hostname = socket.gethostname().lower()
    CS.NOTIFY_TELEGRAM = dist_conf.feature_functions["telegram_notify"]
    CS.USE_MINIFIED_DATA = dist_conf.feature_functions["minify"]

    # print(dist_conf.feature_functions["hosts"].keys())

    CS.FEATURES_TO_GENERATE_MP = dist_conf.feature_functions["hosts"][hostname]["mp"]
    CS.FEATURES_TO_GENERATE_MONO = dist_conf.feature_functions["hosts"][hostname]["mono"]
    CS.SPACY_FUNCTIONS = dist_conf.feature_functions["hosts"][hostname]["spacy"]
    CS.DO_TOPIC_MODELLING = dist_conf.feature_functions["hosts"][hostname]["topic"]
    
    CS.LOAD_COMMENTS = get_judgement_labels in [
        item for sublist in CS.FEATURES_TO_GENERATE_MP["reactions"]+CS.FEATURES_TO_GENERATE_MONO["reactions"] for item in sublist]
    CS.LOAD_FOUNDATIONS = dist_conf.feature_functions["hosts"][hostname]["foundations"]
    CS.LOAD_LIWC = dist_conf.feature_functions["hosts"][hostname]["liwc"]


def create_features():
    """ Iterate over all posts and create features dataframe 

    """
    lg.info("Title as standalone" if CS.TITLE_AS_STANDALONE else "Title Prepended")
    df_posts = globals_loader.df_posts
    df_posts_split = np.array_split(df_posts, CS.NR_THREADS)
    
    # Check location of features...these could just be set or change it in porcess_helpers to get columns by names instead of id.🤷
    #for df in df_posts_split:
    #    feat_lst = list(df.columns)
    #    if feat_lst.index("post_text") != CS.POST_TEXT:
    #        raise ValueError("index of 'post_text' in df_posts is not at CS.POST_TEXT location.")
    #    elif feat_lst.index("post_title") != CS.POST_TITLE:
    #        raise ValueError("index of 'post_title' in df_posts is not at CS.POST_TITLE location.")
    #    elif feat_lst.index("post_id") != CS.POST_ID:
    #        raise ValueError("index of 'post_id' in df_posts is not at CS.POST_ID location.")
    #    elif feat_lst.index("post_author") != CS.POST_AUTHOR:
    #        raise ValueError("index of 'post_author' in df_posts is not at CS.POST_AUTHOR location.")
        

    # setup output dir
    Path(CS.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Do multiprocessing
    processes = []
    q = Queue()

    for i in range(0, CS.NR_THREADS):
        sub_post = df_posts_split[i]
        #p = p_process.parallel_process(q, i, sub_post, features_to_generate, globals_loader.stz_nlp )
        p = p_process.parallel_process(
            q, i, sub_post, CS.FEATURES_TO_GENERATE_MP)
        p.start()
        processes.append(p)

    # Consume queue content as it comes (avoids deadlock)
    multi_feature_df_list = []
    while True:
        try:
            result = q.get(False, 0.01)
            multi_feature_df_list.append(result)
        except queue.Empty:
            pass
        allExited = True
        for t in processes:
            if t.exitcode is None:
                allExited = False
                break
        if allExited & q.empty():
            break

    print("df list"+str(len(multi_feature_df_list)))
    feature_df_multi = pd.concat(multi_feature_df_list, axis=0, join="inner")
    lg.info("Finished Multiprocessing")

    # Do mono processing
    mono_feat_df_list = process_run(
        CS.FEATURES_TO_GENERATE_MONO, df_posts, CS.MONO_ID)
    feature_df_mono = pd.concat(mono_feat_df_list, axis=1, join="inner")
    lg.info("Finished Monoprocessing")

    # Merge mono, multiprocessing and topics
    feature_df = feature_df_multi.merge(
        feature_df_mono, left_on="post_id", right_on="post_id", validate="1:1")

    # Do topic modeling & merge
    if CS.DO_TOPIC_MODELLING:
        if CS.NOTIFY_TELEGRAM:
            msg = "Start {fn} on {host}".format(fn ="topic modelling", host=socket.gethostname())
            sent_telegram_notification(msg)
            
        posts_raw = df_posts["post_text"].to_list()
        post_ids = df_posts["post_id"].to_list()
        topic_df = topic_modelling(posts_raw, post_ids)
        feature_df = topic_df.merge(
            feature_df, left_on="post_id", right_on="post_id", validate="1:1")
        

    # Merge generate features with LIWC & Moral foundations
    if CS.LOAD_LIWC:
        if CS.NOTIFY_TELEGRAM:
           msg = "Start {fn} on {host}".format(fn ="LIWC merge", host=socket.gethostname())
           sent_telegram_notification(msg)
            
        #feature_df = feature_df.merge(
        #    globals_loader.df_liwc, left_on="post_id", right_on=CS.LIWC_PREFIX+"post_id", validate="1:1")
        
        if CS.TITLE_AS_STANDALONE:
            # load post liwc
            feature_df = feature_df.merge(
                globals_loader.df_liwc, left_on="post_id", right_on=CS.LIWC_PREFIX+"post_id", validate="1:1", suffixes=("", "_posts"))
            # load title liwc
            feature_df = feature_df.merge(
                globals_loader.df_liwc_title, left_on="post_id", right_on=CS.LIWC_TITLE_PREFIX+"post_id", validate="1:1", suffixes=("", "_title"))
        else:
            # load merged liwc
            feature_df = feature_df.merge(
                globals_loader.df_liwc_merged, left_on="post_id", right_on=CS.LIWC_MERGED_PREFIX+"post_id", validate="1:1")
            
            
           
    if CS.LOAD_FOUNDATIONS:
        if CS.NOTIFY_TELEGRAM:
           msg = "Started {fn} on {host}".format(fn ="Foundations merge", host=socket.gethostname())
           sent_telegram_notification(msg)
         
         
        if CS.TITLE_AS_STANDALONE:
            # load post liwc
            feature_df = feature_df.merge(
                globals_loader.df_foundations, left_on="post_id", right_on=CS.FOUNDATIONS_PREFIX+"post_id", validate="1:1", suffixes=("", "_posts"))
            # load title liwc
            feature_df = feature_df.merge(
                globals_loader.df_foundations_title, left_on="post_id", right_on=CS.FOUNDATIONS_TITLE_PREFIX+"post_id", validate="1:1", suffixes=("", "_title"))
        else:
            # load merged liwc
            feature_df = feature_df.merge(
                globals_loader.df_foundations_merged, left_on="post_id", right_on=CS.FOUNDATIONS_MERGED_PREFIX+"post_id", validate="1:1")
               

        if CS.NOTIFY_TELEGRAM:
           msg = "Finished {fn} on {host}".format(fn ="Foundations merge", host=socket.gethostname())
           sent_telegram_notification(msg)

    
    if CS.NOTIFY_TELEGRAM:
        msg = "Finished all feature extractions on {host}".format(host=socket.gethostname())
        sent_telegram_notification(msg)
    
    df_size = feature_df.shape[0]
    feature_df = feature_df[~feature_df['post_flair'].str.lower().isin(CS.FLAIRS_TO_REMOVE)]
    lg.info(f"Removed {df_size-feature_df.shape[0]} posts with flairs")
    # Save features in one big dataframe
    date_time = get_date_str()
    feature_df_to_save = feature_df.drop("post_text", axis=1)
    feature_df_to_save = feature_df.drop("post_flair", axis=1)
    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    feature_df_to_save.to_csv(
        CS.OUTPUT_DIR+"features_output_"+date_time+mini+".csv", index=False)

    # Create histogram and sample texts as png
    vis.generate_report(feature_df)


def thread_print():
    """Print the number of threads we will be using
    """
    lg_str = "Using {0} threads".format(CS.NR_THREADS)
    if CS.NR_THREADS < 2:
        lg.warning(lg_str[:-1]+"!")
    else:
        lg.info(lg_str)


def setup(reddit_instance_idx=0):
    """Setup gloabal variables, profanity and print current number of threads

    Args:
        reddit_instance_idx (int, optional): Index of reddit settings we should use for the praw instance. Defaults to 0.
    """
    CS.REDDIT_INSTANCE_IDX = reddit_instance_idx
    globals_loader.init()
    profanity.load_censor_words()
    thread_print()

def refresh_token(gdrive_json):
    """Update bearer token such that we can upload the output zip to gdrive. Othwerise, credentials would have expired

    Args:
        gdrive_json (dict): dictionary of gdrive secrets

    Returns:
        token (string): new bearer token
    """
    token = ""
    client_id = gdrive_json["client_id"]
    client_secret = gdrive_json["client_secret"]
    refresh_token = gdrive_json["refresh_token"]
    refresh_token_cmd = "curl https://www.googleapis.com/oauth2/v4/token \
                            -d client_id={id} \
                            -d client_secret={secret} \
                            -d refresh_token={refresh} \
                            -d grant_type=refresh_token".format(id=client_id, secret=client_secret, refresh=refresh_token)
                            
    p = str(subprocess.check_output(refresh_token_cmd, shell=True)).replace('\\n', '\n').replace('\\t', '\t')
    start_idx = p.find("{")
    end_idx = p.find("}")
    if start_idx != -1 and end_idx != -1:
        p = p[start_idx:end_idx+1]
        dict = json.loads(p)
        token = dict["access_token"]
    return token
    
    
def upload_output_dir():
    """Uploads generated output to personal google drive folder. Only used when running distributed
    """
    hostname = socket.gethostname()
    filename = "{date}_{hostname}_{title}".format(hostname=hostname, date=get_date_str(
        True), title=("standalone" if CS.CS.TITLE_AS_STANDALONE else "prepend"))
    lg.info("Generating zip in output dir: {0}".format(CS.OUTPUT_DIR_ZIPS+filename))
    shutil.make_archive(CS.OUTPUT_DIR_ZIPS+filename, 'zip', CS.OUTPUT_DIR)
    
    path = sys.path[0]+"/secrets/gdrive.json"
    file = open(path)
    data = json.load(file)
    bearer = refresh_token(data)
    folder_id = data["folder_id"]
    link = data["link"]
    upload_to_gdrive_cmd = "curl -X POST -L \
            -H \"Authorization: Bearer {bearer}\" \
            -F \"metadata={{name : '{filename}', parents: ['{folder_id}'] }};type=application/json;charset=UTF-8\" \
            -F \"file=@{file};type=application/zip\" \
            \"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart\"".format(filename=filename, bearer=bearer, folder_id=folder_id, file=CS.OUTPUT_DIR_ZIPS+filename+".zip")
        
    lg.info("Uploading zip to Google drive")  
    p = str(subprocess.check_output(upload_to_gdrive_cmd, shell=True)).replace('\\n', '\n').replace('\\t', '\t')
    if len(p) > 3:
        print(p)
    
    
    if CS.NOTIFY_TELEGRAM:
        msg = "Uploaded {filename} from {host} to GDrive.\n\n{function_list_str}\n{link}".format(
            filename=filename, host=socket.gethostname(), link=link, function_list_str=dist_conf_to_function_list_str(dist_conf))
        
        sent_telegram_notification(msg)


def main(args):
    if "-vis" in args:
        print(CS.OUTPUT_DIR)
        filenames = next(walk(CS.OUTPUT_DIR), (None, None, []))[2] #[] if no file
        #filenames = list(
        #    filter(lambda x: ".csv" in x and "features_output" in x, filenames))
        filenames = list(
            filter(lambda x: ".csv" in x , filenames))
        filenames = sorted(filenames, reverse=True)

        lg.info("Only generating report from "+filenames[0])
        df = pd.read_csv(CS.OUTPUT_DIR+filenames[0], index_col=False)
        globals_loader.load_posts()
        df_posts = globals_loader.df_posts[["post_id", "post_text"]]
        df = pd.merge(df, df_posts, on=['post_id', 'post_id'])


        vis.generate_report(df)
    elif "-dist" in args or "-d" in args:
        # get reddit instance idx
        hostname = socket.gethostname().lower()
        if "reddit_instance_idx" in dist_conf.feature_functions["hosts"][hostname]:
            reddit_instance_idx = dist_conf.feature_functions["hosts"][hostname]["reddit_instance_idx"]
            setup(reddit_instance_idx)
        
        # If we run it in a distributed manner, we might run it once for title standalone once prepend
        lg.info("Running distributed.")
        title_handling = dist_conf.feature_functions["title_handling"]
        if title_handling > 1:
            for title_as_standalone in [False, True]:
                try:
                    set_features_to_run_dist(title_as_standalone)
                    #setup()
                    create_features()
                    upload_output_dir()
                except Exception as e:
                    print(sys.exc_info()[2])
                    sent_telegram_notification("Crashed on {hostname}\n    {e_str}".format(e_str=str(e), hostname=hostname))
        else:
            try:
                set_features_to_run_dist(bool(title_handling))
                #setup()
                create_features()
                upload_output_dir()
            except Exception as e:
                lg.exception(e)
                sent_telegram_notification("Crashed on {hostname}\n    {e_str}".format(e_str=str(e), hostname=hostname))
            
    elif "-upload" in args or "-u" in args:
        upload_output_dir()
    else:
        setup()
        create_features()
        upload_output_dir()
        


if __name__ == "__main__":
    main(sys.argv)
