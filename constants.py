"""
Constants.py
"""
import os
import multiprocessing
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *

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
NR_THREADS =  multiprocessing.cpu_count()
TMP_SAVE_DIR = OUTPUT_DIR+"feature_df_tmp"
MONO_ID = "mono"

# FEATURE DF POSTPEND
POST_PEND = ["post_id", "post_text"]
POST_PEND_MONO = ["post_id"]

# PERCENTAGE TO MINIFY POSTS
MINIFY_FRAC = 0.1

# STANZA STRINGS
SP_VERB = "VERB"

#SP_VOICE_ACTIVE = "Act"
#SP_VOICE_PASSIVE = "nsubjpass"

SP_TENSE_PAST = "Past"
SP_TENSE_PRESENT = "Pres"
SP_TENSE_FUTURE = "Fut"

SP_FEATS_TENSE = "Tense"
SP_FEATS_VOICE = "Voice"


#Feature Generation
FEATURES_TO_GENERATE_MP = {
    "speaker":[
        #(get_author_amita_post_activity, CS.POST_AUTHOR),
        #(get_author_age, CS.POST_AUTHOR),
        #(get_post_author_karma, CS.POST_AUTHOR)
    ], 
    "writing_sty":[
        (get_punctuation_count, CS.POST_TEXT),
        #(get_tense_time_and_voice, CS.POST_TEXT)
    ],
    "behaviour":[
    ],
    "reactions":[
        #(get_judgement_labels, CS.POST_ID)
    ]
}

FEATURES_TO_GENERATE_MONO = {
    "speaker":[
        #(get_author_amita_post_activity, CS.POST_AUTHOR),
        #(get_author_age, CS.POST_AUTHOR),
        #(get_post_author_karma, CS.POST_AUTHOR)
    ], 
    "writing_sty":[
        #(get_punctuation_count, CS.POST_TEXT),
        (get_tense_time_and_voice, CS.POST_TEXT)
    ],
    "behaviour":[
    ],
    "reactions":[
        #(get_judgement_labels, CS.POST_ID)
    ]
}