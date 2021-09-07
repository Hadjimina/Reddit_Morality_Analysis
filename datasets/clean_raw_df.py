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

raw_csv_name  = "posts_raw_27_6_2021.csv"
full_path = CS.dataset_dir+raw_csv_name
lg.info("Cleaning {0} at {1}".format(raw_csv_name, full_path))

# read csv 
df = pd.read_csv(full_path)
orig_length = df.shape[0]

# Drop NANs
df = df.dropna()
after_na_drop = df.shape[0]

# Remove strings
str_to_remove = ["[removed]", "[deleted]"]
cols_to_clean = ["post_text", "post_title", "post_author_id"]
for col in cols_to_clean:
    for to_remove in str_to_remove:
        df = df[df[col] != to_remove]
after_string_drop = df.shape[0]

# Print dropped info
lg.info("Dropped {0} Nans, {1} bad strings => {2} remaining".format(orig_length-after_na_drop, after_na_drop-after_string_drop, after_string_drop))

# save
save_name = raw_csv_name.replace("raw", "cleaned")
full_path = CS.dataset_dir+save_name
lg.info("Saving {0} to {1}".format(save_name, full_path))
df.to_csv(full_path, index=False)