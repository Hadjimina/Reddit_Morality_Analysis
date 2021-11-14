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
from helpers.requirements import *

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

    # fix "uniquely valued index object"
    for df in multi_feature_df_list:
        df.reset_index(inplace=True, drop=True)
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
        posts_raw = df_posts["post_text"].to_list()
        post_ids = df_posts["post_id"].to_list()
        topic_df = topic_modelling(posts_raw, post_ids)
        feature_df = topic_df.merge(
            feature_df, left_on="post_id", right_on="post_id", validate="1:1")

    # Merge generate features with LIWC & Moral foundations
    if CS.LOAD_LIWC:
        feature_df = feature_df.merge(
            globals_loader.df_liwc, left_on="post_id", right_on=CS.LIWC_PREFIX+"post_id", validate="1:1")
        if CS.TITLE_AS_STANDALONE:
            feature_df = feature_df.merge(
                globals_loader.df_liwc_title, left_on="post_id", right_on=CS.LIWC_TITLE_PREFIX+"post_id", validate="1:1")
    if CS.LOAD_FOUNDATIONS:
        feature_df = feature_df.merge(globals_loader.df_foundations, left_on="post_id",
                                      right_on=CS.FOUNDATIONS_PREFIX+"post_id", validate="1:1")
        if CS.TITLE_AS_STANDALONE:
            feature_df = feature_df.merge(globals_loader.df_foundations_title,
                                          left_on="post_id", right_on=CS.FOUNDATIONS_TITLE+"post_id", validate="1:1")

    # Save features in one big dataframe
    date_time = get_date_str()
    feature_df_to_save = feature_df.drop("post_text", axis=1)
    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    feature_df_to_save.to_csv(
        CS.OUTPUT_DIR+"features_output_"+date_time+mini+".csv", index=False)

    # Enforce minimum post requiremnets
    if CS.ENFORCE_POST_REQUIREMENTS:
        feature_df = enforce_requirements(feature_df)

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


def setup():
    """Load globals and print number of threads used, before running "create_features"
    """
    globals_loader.init()
    profanity.load_censor_words()
    thread_print()

def refresh_token(gdrive_json):
    """Update bearer token such that we can upload the output zip to gdrive. Othwerise, credentials would have expired

    Args:
        gdrive_json (dict): dictionary of gdrive secrets
    """
    client_id = gdrive_json["client_id"]
    client_secret = gdrive_json["client_secret"]
    refresh_token = gdrive_json["refresh_token"]
    refresh_token_cmd = "curl https://www.googleapis.com/oauth2/v4/token \
                            -d client_id={id} \
                            -d client_secret={secret} \
                            -d refresh_token={refresh} \
                            -d grant_type=refresh_token".format(id=client_id, secret=client_secret, refresh=refresh_token)
    p = subprocess.Popen(refresh_token_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
    
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
    refresh_token(data)
    
    bearer = data["bearer"]
    folder_id = data["folder_id"]
    link = data["link"]
    upload_to_gdrive_cmd = "curl -X POST -L \
            -H \"Authorization: Bearer {bearer}\" \
            -F \"metadata={{name : '{filename}', parents: ['{folder_id}'] }};type=application/json;charset=UTF-8\" \
            -F \"file=@{file};type=application/zip\" \
            \"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart\"".format(filename=filename, bearer=bearer, folder_id=folder_id, file=CS.OUTPUT_DIR_ZIPS+filename+".zip")
        
    lg.info("Uploading zip to Google drive")  
    p = subprocess.Popen(upload_to_gdrive_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
    if CS.NOTIFY_TELEGRAM:
        msg = "Uploaded {filename} from {host} to GDrive.\n    {link}".format(filename=filename, host=socket.gethostname(), link=link)
        sent_telegram_notification(msg)


def main(args):
    if "-vis" in args:
        filenames = next(walk(CS.OUTPUT_DIR), (None, None, []))[
            2]  # [] if no file
        filenames = list(
            filter(lambda x: ".csv" in x and "features_output" in x, filenames))
        filenames = sorted(filenames, reverse=True)

        lg.info("Only generating report from "+filenames[0])
        df = pd.read_csv(CS.OUTPUT_DIR+filenames[0], index_col=False)
        globals_loader.load_posts()
        df_posts = globals_loader.df_posts[["post_id", "post_text"]]
        df = pd.merge(df, df_posts, on=['post_id', 'post_id'])

        # Enforce minimum post requiremnets
        if CS.ENFORCE_POST_REQUIREMENTS:
            df = enforce_requirements(df)

        vis.generate_report(df)
    elif "-dist" in args or "-d" in args:
        setup()
        # If we run it in a distributed manner, we might run it once for title standalone once prepend
        lg.info("Running distributed.")
        title_handling = dist_conf.feature_functions["title_handling"]
        if title_handling > 1:
            for title_as_standalone in [False, True]:
                set_features_to_run_dist(title_as_standalone)
                create_features()
                upload_output_dir()
        else:
            set_features_to_run_dist(bool(title_handling))
            create_features()
            upload_output_dir()
    else:
        setup()
        create_features()
        upload_output_dir()
        


if __name__ == "__main__":
    main(sys.argv)
