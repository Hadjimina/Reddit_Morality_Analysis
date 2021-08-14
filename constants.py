"""
Constants.py
"""
import os

USE_MINIFIED_DATA = False

#directories
dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/datasets/data/"

HOME_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))+"/output/"

POSTS_RAW = dataset_dir+"posts_27_6_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"posts_27_6_2021.csv"
POSTS_CLEAN = dataset_dir+"posts_cleaned_27_6_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"posts_cleaned_27_6_2021.csv"
COMMENTS_RAW = dataset_dir+"comments_16_07_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"comments_16_07_2021.csv"


# COLUMN INDICES
#INDEX = 0
POST_ID = 0
POST_TEXT = 1
POST_TITLE = 2
POST_AUTHOR = 3

# VISUALISATION
DIAGRAM_HIDE_0_VALUES = True

# JUDGEMENT LABELS
JUDGMENT_ACRONYM = ["YTA", "NTA", "INFO", "ESH", "NAH"]
JUDGMENT_LABEL = ["You're the Asshole", "Not the Asshole", "Everyone Sucks here", "No Assholes Here", "Not Enough Info"]

# MULTITHREADING
NR_THREADS = 8
TMP_SAVE_DIR = OUTPUT_DIR+"/feature_df_tmp"

# FEATURE DF POSTPEND
POST_PEND = ["post_id", "post_text"]