"""
Constants.py
"""
import os
dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/datasets/data/"

HOME_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))+"/output/"

POSTS_RAW = dataset_dir+"posts_27_6_2021.csv"
POSTS_CLEAN = dataset_dir+"posts_cleaned_27_6_2021.csv"
COMMENTS_RAW = dataset_dir+"comments_16_7_2021.csv"

INDEX = 0
POST_ID = 0
POST_TEXT = 1
POST_TITLE = 2
POST_AUTHOR = 3

USE_MINIFIED_DATA = True

# VISUALISATION
DIAGRAM_HIDE_0_VALUES = True