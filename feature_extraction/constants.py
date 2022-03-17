"""
Constants.py
"""
import os
from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from feature_functions.topic_modelling import *
import multiprocessing

# Minification, requirements and title
# Modified in create_features.set_featueres_to_run_dist => so no longer const
USE_MINIFIED_DATA = False
TITLE_AS_STANDALONE = False
NOTIFY_TELEGRAM = False
REDDIT_INSTANCE_IDX = 1

# TOPIC MODELING
# TODO: this value was chosen arbitrarily. should it stay a frac or absolute?
#TOPICS_ABS = 300000000000
MIN_CLUSTER_PERC = 0.002

# PERCENTAGE TO MINIFY POSTS
MINIFY_FRAC = 0.05

# Prefixes
LIWC_PREFIX = "liwc_"
LIWC_TITLE_PREFIX = "liwc_title_"
LIWC_MERGED_PREFIX = "liwc_merged_"
FOUNDATIONS_PREFIX = "foundations_"
FOUNDATIONS_TITLE_PREFIX = "foundations_title_"
FOUNDATIONS_MERGED_PREFIX = "foundations_merged_"
TOPIC_PREFIX = "topic_"

# directories
dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/datasets/data/"

HOME_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

OUTPUT_DIR = os.path.dirname(os.path.abspath(
    __file__))+"/output/"+output_dir_name()+"/"
OUTPUT_DIR_ZIPS = os.path.dirname(os.path.abspath(__file__))+"/output/zips/"

mini_str = "_mini" if USE_MINIFIED_DATA else ""
POSTS_RAW = "{0}posts_27_6_2021{1}.csv".format(dataset_dir, mini_str)
POSTS_CLEAN = "{0}posts_cleaned_27_6_2021{1}.csv".format(dataset_dir, mini_str)
COMMENTS_RAW = "{0}comments_raw_16_07_2021{1}.csv".format(
    dataset_dir, mini_str)
COMMENTS_CLEAN = "{0}comments_cleaned_16_07_2021{1}.csv".format(
    dataset_dir, mini_str)
LIWC = "{0}LIWC_27_6_2021{1}.csv".format(dataset_dir, mini_str)
LIWC_TITLE = "{0}LIWC_title_27_6_2021{1}.csv".format(dataset_dir, mini_str)
LIWC_MERGED = "{0}LIWC_merged_27_6_2021{1}.csv".format(dataset_dir, mini_str)
FOUNDATIONS = "{0}moral_foundations_27_6_2021{1}.csv".format(
    dataset_dir, mini_str)
FOUNDATIONS_TITLE = "{0}moral_foundations_title_27_6_2021{1}.csv".format(
    dataset_dir, mini_str)
FOUNDATIONS_MERGED = "{0}moral_foundations_merged_27_6_2021{1}.csv".format(
    dataset_dir, mini_str)

# COLUMN INDICES
#INDEX = 0
POST_ID = "post_id"
POST_TEXT = "post_text"
POST_TITLE = "post_title"
POST_AUTHOR = "post_author"

# VISUALISATION
DIAGRAM_HIDE_0_VALUES = False
NR_COLS_MPL = 5
NR_COLS_TEXT = 10
MAX_FEATURES_TO_DISPLAY = 25

# JUDGEMENT LABELS
JUDGMENT_ACRONYM = ["YTA", "NTA", "INFO", "ESH", "NAH"]
JUDGEJMENT_DICT = {  # everytime asshole appears an expression get added that replaces asshole with "ah"
    "YTA": ["YTA", "You're the asshole", "You are the asshole", "You are a little bit the asshole", "YWBTA", "You are an asshole", "You're an asshole", "yt absolute a"],
    "NTA": ["NTA", "not the asshole", "not an asshole", "don't think you're an asshole", "YWNBTA"],
    "INFO": ["INFO", "Not enough info", "Not enough information", "More info", ],
    "ESH": ["ESH", "everyone sucks here", "everybody sucks here", "ETA", "everyone's the asshole"],
    "NAH": ["NAH", "No Assholes here", "No Asshole here", "no one sucks here"],
}
BOT_STRINGS = ["automod", "i am a bot"]
AITA = ["am i the asshole", "aita", "aitah", "am i an asshole", "aith", "was i the asshole", "was i an asshole", "amia", "am i ta",
        "am i being an asshole", "am i being the asshole", "was I an asshole", "was I the asshole", "amita", "was i ta", "am i ta"]
WIBTA = ["wita", "witah", "wibta", "wibtah",
         "would i be the asshole", "would i really be the asshole"]

# MULTITHREADING
NR_THREADS = multiprocessing.cpu_count()
TMP_SAVE_DIR = OUTPUT_DIR+"feature_df_tmp"
MONO_ID = "mono"

# FEATURE DF POSTPEND
# post_id,post_text,post_title,post_author_id,post_score,post_created_utc,post_num_comments
# "post_ups", "post_downs"
POST_PEND = ["post_id",  "post_num_comments", "post_flair"]  # "post_text",
POST_PEND_MONO = ["post_id"]

# SPACY STRINGS
SP_VERB = "VERB"

SP_TENSE_PAST = "Past"
SP_TENSE_PRESENT = "Pres"
SP_TENSE_FUTURE = "Fut"

SP_FEATS_TENSE = "Tense"
SP_FEATS_VOICE = "Voice"

# list from https://www.lingographics.com/english/personal-pronouns/
PRONOUNS = [["i", "me", "my", "mine", "myself"],
            ["you", "your", "yours", "yourself"],
            ["he", "him", "his", "himself", "she", "her",
                "hers", "herself", "it", "its", "itself"],
            ["we", "us", "our", "ours", "ourselves"],
            ["yourselves"],
            ["they", "them", "their", "theirs", "themselves"]]
PRONOUN_MATCHING_MISC = ["i'm", "i am", "aita", "im"]
FLAIRS_TO_REMOVE = ["troll", "meta", "news", "shitpost", "troll", "community discussion",
                    "awards", "fake", "announcement", "spam", "new rule", "shit post", "Unintelligible"]

# EMO LEX EMOTIONS
EMOTIONS = ['fear', 'anger', 'trust', 'surprise', 'sadness',
            'disgust', 'joy', 'anticipation', 'positive', 'negative']

# Feature Generation

# Modified in create_features.set_featueres_to_run_dist => so no longer const
FEATURES_TO_GENERATE_MP = {
    "speaker": [
        #(get_author_amita_post_activity, CS.POST_AUTHOR),
        #(get_author_info, CS.POST_AUTHOR),

        (get_author_age_and_gender, CS.POST_TEXT)
    ],
    "writing_sty": [
        (get_punctuation_count, CS.POST_TEXT),
        (get_emotions, CS.POST_TEXT),
        (aita_location, CS.POST_TEXT),
        (get_profanity_count, CS.POST_TEXT),
        (check_wibta, CS.POST_TEXT),
        
        (get_spacy_features, CS.POST_TEXT)# remove this for feature generation

    ],
    "reactions": [
        # (check_crossposts, CS.POST_ID),  # slow
        #(get_judgement_labels, CS.POST_ID)
    ]
}

# Modified in create_features.set_featueres_to_run_dist => so no longer const
FEATURES_TO_GENERATE_MONO = {
    "writing_sty": [
        # (get_spacy_features, CS.POST_TEXT),  # => 4h for 10%
    ],
}

# Modified in create_features.set_featueres_to_run_dist => so no longer const...
SPACY_FUNCTIONS = [
    get_tense_in_spacy,
    get_voice_in_spacy,
    get_sentiment_in_spacy,
    get_focus_in_spacy,
    get_emotions_self_vs_other_in_spacy,
    get_profanity_self_vs_other_in_spacy,
]

# Modified in create_features.set_featueres_to_run_dist => so no longer const...
DO_TOPIC_MODELLING = False
TOPIC_DOWNSAMPLE = False
TOPIC_DOWNSAMPLE_FRAC = 0.3

# Loading
# Modified in create_features.set_featueres_to_run_dist => so no longer const...
LOAD_POSTS = True
# get_judgement_labels in [item for sublist in FEATURES_TO_GENERATE_MP["reactions"]+FEATURES_TO_GENERATE_MONO["reactions"] for item in sublist]
LOAD_COMMENTS = True
LOAD_FOUNDATIONS = False
LOAD_LIWC = False
