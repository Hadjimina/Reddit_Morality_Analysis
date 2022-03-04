
import sys
import os
import spacy
import pandas as pd
import re
import liwc
import xgboost as xgb
from collections import Counter
from flask import Flask, render_template, request

app = Flask(__name__)

p = os.path.abspath('../../feature_extraction')
sys.path.insert(1, p)


from feature_functions.topic_modelling import *
from feature_functions.writing_style_features import get_spacy_features, get_punctuation_count, get_emotions, aita_location, get_profanity_count, check_wibta
from feature_functions.speaker_features import get_author_age_and_gender

# TODO if we deploy this as an actual website. Create copies of these functions and create copy repo
# ALSO FIX constants file

LIWC_PATH = "./data/liwc.dic"
MF_PATH = "./data/mf.dic"
MF_GAP = "                    "
XGB_PATH = "./data/xgboost03.03.2022.json"
RF_PATH = "./data/rf03.03.2022.json"
TRAIN_CSV = "./data/prepend_done_trained_feats.csv"
JUDGMENT_ACRONYM = ["YTA", "NTA", "INFO", "ESH", "NAH"]



def loadSpacy():
    """Setup the spacy pipeline
    """
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("spacytextblob")
    return nlp


def runSpeakerFeatures(post_text):
    tupleList = []
    tupleList = tupleList + get_author_age_and_gender(post_text)
    return tupleList


def runStyleFeatures(post_text):
    tupleList = []
    nlp = loadSpacy()
    tupleList = tupleList + get_punctuation_count(post_text, nlp)
    tupleList = tupleList + get_emotions(post_text)
    tupleList = tupleList + aita_location(post_text)
    tupleList = tupleList + get_profanity_count(post_text)
    tupleList = tupleList + check_wibta(post_text, nlp)
    tupleList = tupleList + get_spacy_features(post_text, nlp)

    # Prefix all tuples with "writing_sty_" if necessary
    prefix = "writing_sty_"
    for i in list(range(len(tupleList))):
        if not tupleList[i][0].startswith(prefix):
            tupleList[i] = (prefix+tupleList[i][0], tupleList[i][1])
    return tupleList

def tokenize(text):
    text = text.lower()
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


def analyseLIWC(post_text, dict_path):
    parse, category_names = liwc.load_token_parser(dict_path)
    result = Counter(category for token in tokenize(post_text)
                     for category in parse(token))
    result_dict = dict(result)
    print(result_dict)
    #result_val = list(map(lambda x: [x], list(result_dict.values())))
    # modify keys
    if dict_path == LIWC_PATH:
        result_key = list(
            map(lambda x: "liwc_"+x.split("(")[0][:-1], list(result_dict.keys())))
    else:
        result_key = list(map(lambda x: "foundations_" +
                          x.split("\t")[0]+MF_GAP+x.split("\t")[1], list(result_dict.keys())))


    result_dict = zip(result_key, list(result_dict.values()))
    # print(category_names)
    # print(result_dict)
    #df = pd.DataFrame.from_dict(result_dict)
    # print(df)
    print(dict(result_dict))
    return result_dict


def dictValToList(feat_dict):
    result_val = list(map(lambda x: [x], list(feat_dict.values())))
    return dict(zip(list(feat_dict.keys()), result_val))


def reorderColumns(df):
    # Reorder the columns as in the sample training df provided. We need this b.c. order of features need to be same as in training for prediction to work
    df_train = pd.read_csv(TRAIN_CSV, nrows=2)
    vote_acro_feats = list(
        filter(lambda x: any(substring in x for substring in JUDGMENT_ACRONYM), list(df_train.columns)))
    df_train = df_train.drop(vote_acro_feats, axis=1)
    feats_train = list(df_train.columns)
    feats_pred = list(df.columns)

    len_train = len(feats_train)
    len_pred = len(feats_pred)

    if not len_train == len_pred:
        print(f"missing features\n{set(feats_train)-set(feats_pred)}")
        raise Exception(
            f"Dataframe length mismatch. df_train = {len_train}, df_pred={len_pred}")
    elif not set(feats_train) == set(feats_pred):
        raise Exception(
            f"df_pred does not contain features:\n{set(feats_train)-set(feats_pred)}")

    return df[df_train.columns]


def getFeatureValues(post_text):
    all_features = []
    # LIWC
    # not all categories returned?
    #all_features += analyseLIWC(post_text, LIWC_PATH)
    all_features +=analyseLIWC(post_text, MF_PATH)#Somehow not working, dict probably strang spacing
    all_features += runSpeakerFeatures(post_text)
    all_features += runStyleFeatures(post_text)
    all_features_dict = dictValToList(dict(all_features))
    df = pd.DataFrame.from_dict(all_features_dict)
    return df


def getPrediction(df):
    df_ordered = reorderColumns(df)
    clf = xgb.XGBRegressor()
    clf.load_model(XGB_PATH)

    y_pred = clf.predict(df_ordered)
    return y_pred


@app.route('/', methods=['GET', 'POST'])
def index():
    data = []
    if request.method == 'POST':
        if request.form['submit_posts'] == 'Analyze':

            df_old = getFeatureValues(request.form['old_post'])
            y_old = getPrediction(df_old)

            df_new = getFeatureValues(request.form['new_post'])
            y_new = getPrediction(df_new)

            data = [
                {
                    "ahr_old": y_old,
                    "ahr_new": y_new,
                    "changedFeatures": [
                        {"name": "feat1",
                         "value": 2,
                         "perc_change": 0.1},
                        {"name": "feat2",
                         "value": 3,
                         "perc_change": -0.1},
                    ]
                }
            ]
        else:
            pass  # unknown
    return render_template('index.html', data=data)


app.run(host='0.0.0.0', port=3001)
