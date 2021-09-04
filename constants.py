"""
Constants.py
"""
import os
import multiprocessing



USE_MINIFIED_DATA = True
LOAD_COMMENTS = False

#directories
dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/datasets/data/"

HOME_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))+"/output/"

POSTS_RAW = dataset_dir+"posts_27_6_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"posts_27_6_2021.csv"
POSTS_CLEAN = dataset_dir+"posts_cleaned_27_6_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"posts_cleaned_27_6_2021.csv"
COMMENTS_RAW = dataset_dir+"comments_16_07_2021.csv" #dataset_dir+"comments_16_07_2021_mini.csv" if USE_MINIFIED_DATA else dataset_dir+"comments_16_07_2021.csv"


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
NR_THREADS = 1# multiprocessing.cpu_count()
TMP_SAVE_DIR = OUTPUT_DIR+"/feature_df_tmp"

# FEATURE DF POSTPEND
POST_PEND = ["post_id", "post_text"]

# PERCENTAGE TO MINIFY POSTS
MINIFY_FRAC = 0.1

# STANZA STRINGS
ST_VERB = "VERB"

ST_VOICE_ACTIVE = "Act"
ST_VOICE_PASSIVE = "Pass"

ST_TENSE_PAST = "Past"
ST_TENSE_PRESENT = "Pres"
ST_TENSE_FUTURE = "Fut"

ST_FEATS_TENSE = "Tense"
ST_FEATS_VOICE = "Voice"
