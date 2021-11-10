"""
    Simple script to clean a raw dataframe which was created with the dataset/create_datasets/get_posts.py script
"""
import logging as lg
import coloredlogs
coloredlogs.install()
import pandas as pd
import sys 
sys.path.append('..')
import constants as CS

def clean(posts=True):
    raw_csv_name  = "posts_raw_27_6_2021.csv" if posts else "comments_raw_16_07_2021.csv"
    full_path = CS.dataset_dir+raw_csv_name
    lg.info("Cleaning {0} at {1}".format(raw_csv_name, full_path))

    # read csv 
    df = pd.read_csv(full_path)
    orig_length = df.shape[0]

    # Drop NANs
    df = df.dropna()
    after_na_drop = df.shape[0]

    # Remove strings
    entries_to_remove = ["[removed]", "[deleted]"]
    cols_to_clean = ["post_text", "post_title"] if posts else ["comment_text"]
    strs_to_clean = ["i am a bot"] # Mainly used for comment filtering
    for col in cols_to_clean:
        for entry_to_remove in entries_to_remove:
            df = df[df[col] != entry_to_remove]
        for str_to_remove in strs_to_clean:
            df = df[~df[col].str.lower().contains(str_to_remove)]
    after_string_drop = df.shape[0]

    # remove flair
    if "posts" in raw_csv_name:
        #TODO: remove all posts with certain flair
        a = 1
        
    # Print dropped info
    lg.info("Dropped {0} Nans, {1} bad stringsm from {3} => {2} remaining".format(orig_length-after_na_drop, after_na_drop-after_string_drop, after_string_drop, raw_csv_name))

    # save
    save_name = raw_csv_name.replace("raw", "cleaned")
    full_path = CS.dataset_dir+save_name
    lg.info("Saving {0} to {1}".format(save_name, full_path))
    df.to_csv(full_path, index=False)

if __name__ == "__main__":
    
    if "-comments" in sys.argv or "-com" in sys.argv:
        clean(posts=False)
    elif "-posts" in sys.argv:
        clean()
    else:
        clean()
        clean(posts=False)