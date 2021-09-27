"""
Constants.py
"""
import os
import multiprocessing
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from feature_functions.behaviour_features import *

ENFORCE_POST_REQUIREMENTS = True

# Loading & minified
USE_MINIFIED_DATA = True
LOAD_COMMENTS = True
LOAD_FOUNDATIONS = False
LOAD_LIWC = False

# Prefixes
LIWC_PREFIX = "liwc_"
FOUNDATIONS_PREFIX = "foundations_"

#directories
dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/datasets/data/"

HOME_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))+"/output/"

mini_str = "_mini" if USE_MINIFIED_DATA else ""
POSTS_RAW = "{0}posts_27_6_2021{1}.csv".format(dataset_dir, mini_str)
POSTS_CLEAN = "{0}posts_cleaned_27_6_2021{1}.csv".format(dataset_dir, mini_str)
COMMENTS_RAW = "{0}comments_16_07_2021{1}.csv".format(dataset_dir, mini_str)
LIWC = "{0}LIWC_fixed_27_6_2021{1}.csv".format(dataset_dir, mini_str)
FOUNDATIONS = "{0}moral_foundations_fixed_27_6_2021{1}.csv".format(dataset_dir, mini_str)

# COLUMN INDICES
#INDEX = 0
POST_ID = 0
POST_TEXT = 1
POST_TITLE = 2
POST_AUTHOR = 3

# VISUALISATION
DIAGRAM_HIDE_0_VALUES = True
NR_COLS_MPL = 5
NR_COLS_TEXT = 10
MAX_FEATURES_TO_DISPLAY = 25

# JUDGEMENT LABELS
JUDGMENT_ACRONYM = ["YTA", "NTA", "INFO", "ESH", "NAH"]
JUDGMENT_LABEL = ["You're the Asshole", "Not the Asshole", "Everyone Sucks here", "No Assholes Here", "Not Enough Info"]

# MULTITHREADING
NR_THREADS =  1#multiprocessing.cpu_count()
TMP_SAVE_DIR = OUTPUT_DIR+"feature_df_tmp"
MONO_ID = "mono"

# FEATURE DF POSTPEND
POST_PEND = ["post_id", "post_text"]
POST_PEND_MONO = ["post_id"]

# PERCENTAGE TO MINIFY POSTS
MINIFY_FRAC = 0.001

# SPACY STRINGS
SP_VERB = "VERB"

SP_TENSE_PAST = "Past"
SP_TENSE_PRESENT = "Pres"
SP_TENSE_FUTURE = "Fut"

SP_FEATS_TENSE = "Tense"
SP_FEATS_VOICE = "Voice"

# list from https://www.lingographics.com/english/personal-pronouns/
PRONOUNS = [["i", "me", "my", "mine", "myself"],
            ["you", "your","yours", "yourself"], 
            ["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"],
            ["we", "us", "our", "ours", "ourselves"], 
            ["yourselves"], # TODO: 2nd Person plurar & singular is almost identical
            ["they", "them", "their", "theirs", "themselves"]]
PRONOUN_AGE_GENDER_DIST = 3 # max number of characters that may lie between end of pronoun and start of age/gender description. e.g. "My (23M) boyfriend" => 2 characters

# EMO LEX EMOTIONS
EMOTIONS = ['fear', 'anger', 'anticip', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation']

# TOPIC MODELING
NR_TOPICS_PER_POST = 10

#Feature Generation
FEATURES_TO_GENERATE_MP = {
    "speaker":[
        #(get_author_amita_post_activity, CS.POST_AUTHOR),
        #(get_author_info, CS.POST_AUTHOR), # => seems to be ultra slow!!!
        #(get_author_account_age, CS.POST_AUTHOR),
        #(get_post_author_karma, CS.POST_AUTHOR),
        #(get_author_age_and_gender, CS.POST_TEXT)
    ], 
    "writing_sty":[
        #(get_punctuation_count, CS.POST_TEXT),
        #(get_emotions, CS.POST_TEXT),
        #(aita_location, CS.POST_TEXT),
        #(get_profanity_count, CS.POST_TEXT),
        
    ],
    "behaviour":[
        #(get_N_topics, CS.POST_TEXT)
    ],
    "reactions":[
        #(get_judgement_labels, CS.POST_ID)
    ]
}

FEATURES_TO_GENERATE_MONO = {
    "speaker":[
    ], 
    "writing_sty":[
        #(get_spacy_features, CS.POST_TEXT), # => 4h for 10%
    ],
    "behaviour":[
    ],
    "reactions":[
    ]
}

SPACY_FUNCTIONS = [  #get_tense_in_spacy,
                     #get_voice_in_spacy,
                     #get_sentiment_in_spacy, 
                     #get_focus_in_spacy, 
                     #get_emotions_self_vs_other_in_spacy,
                     #get_profanity_self_vs_other_in_spacy,
                     ]