import helpers.globals_loader as globals_loader
from feature_functions.writing_style_features import *
import spacy
import coloredlogs
import logging as lg
import pandas as pd


coloredlogs.install()
globals_loader.init()



df_posts = globals_loader.df_posts
nlp = spacy.load('en_core_web_trf')
column = df_posts["post_text"]

for index, value in column.items():
    get_tense_time_and_voice(value, nlp)
#temp_s = column.apply(get_tense_time_and_voice, stz_nlp=stz_nlp)