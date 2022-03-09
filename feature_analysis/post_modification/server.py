

from functools import reduce
import sys
import os
import spacy
import pandas as pd
import re
import xgboost as xgb
from collections import Counter
from flask import Flask, render_template, request, url_for
import subprocess
import json

app = Flask(__name__)

p = os.path.abspath('../../feature_extraction')
sys.path.insert(1, p)

from feature_functions.speaker_features import get_author_age_and_gender
from feature_functions.writing_style_features import get_spacy_features, get_punctuation_count, get_emotions, aita_location, get_profanity_count, check_wibta
from feature_functions.topic_modelling import *
from helpers.process_helper import process_run
import constants as CS
from helpers.globals_loader import load_spacy

LIWC_EXE_PATH = "LIWC-22-cli"
LIWC_IN = "./post_to_analyse.csv"
MF_PATH = "./data/mf.dic"
LIWC_2015 = "LIWC2015"

MODEL_ME = "0.33717"
XGB_PATH = "./data/xgboost09.03.2022.json"
#RF_PATH = "./data/rf03.03.2022.json"
TRAIN_CSV = "./data/prepend_done_trained_feats.csv"
JUDGMENT_ACRONYM = ["YTA", "NTA", "INFO", "ESH", "NAH"]

with open('./feature_explanation.json', 'r') as f:
    FEAT_EXPLANATION = json.load(f)


def loadSpacy():
    """Setup the spacy pipeline
    """
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("spacytextblob")
    return nlp


def runSpeakerFeatures(post_text):
    tupleList = []
    tupleList = tupleList + get_author_age_and_gender(post_text)
    tupleList = list(map(lambda x: ("speaker_"+x[0], x[1]), tupleList))
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


def analyseLIWC(csv_input, is_foundations=False):
    # liwc out, hacky
    liwc_out = "./foundations_tmp.csv" if is_foundations else "./liwc_tmp.csv"
    liwcDict = MF_PATH if is_foundations else "LIWC2015"

    call_liwc = [LIWC_EXE_PATH,
                 "--mode", "wc",
                 "--dictionary", MF_PATH if is_foundations else LIWC_2015,
                 "--input", csv_input,
                 "--output", liwc_out]

    subprocess.call(call_liwc)

    df_out = pd.read_csv(liwc_out)
    if "Row ID" in list(df_out.columns):
        df_out = df_out.drop(columns=["Row ID"])
    return df_out


def dictValToList(feat_dict):
    result_val = list(map(lambda x: [x], list(feat_dict.values())))
    return dict(zip(list(feat_dict.keys()), result_val))


def reorderColumns(df):
    # Reorder the columns as in the sample training df provided. We need this b.c. order of features need to be same as in training for prediction to work
    df_train = pd.read_csv(TRAIN_CSV, nrows=2)
    vote_acro_feats = list(
        filter(lambda x: any(substring in x for substring in JUDGMENT_ACRONYM), list(df_train.columns)))
    df_train = df_train.drop(vote_acro_feats, axis=1)
    if "post_id" in list(df_train.columns):
        df_train = df_train.drop(columns=["post_id"])

    feats_train = list(df_train.columns)
    # fix shitty mf columns
    feats_train = list(
        map(lambda x: "foundations_"+x.split("_")[1][2:].strip() if "foundations_" in x and x.split("_")[1][0].isdigit() else x, feats_train))

    feats_pred = list(df.columns)

    s1 = set(feats_train)
    s2 = set(feats_pred)

    if len(s1-s2) > 0:
        print(len(s1))
        print(len(s2))
        print("This should be empty. If note we are not generating the following features in post modificaiton")
        print(s1-s2)

    return df[feats_train]


def getFeatureValues(post_text, is_modified=False):
    load_spacy()
    df_tmp = pd.DataFrame(data={"post_text": [post_text]})
    df_tmp.to_csv(LIWC_IN, index=False)
    feat_type = "mod" if is_modified else "orig"

    # Run through manual features
    feat_df_list = process_run(
        CS.FEATURES_TO_GENERATE_MP, df_tmp, 0)
    feature_df = pd.concat(feat_df_list, axis=1, join="inner")
    feature_df.to_csv(feat_type + "_feat_tmp.csv", index=False)

    # Run LIWC 2015
    liwc_2015 = analyseLIWC(LIWC_IN, is_foundations=False)
    liwc_2015 = liwc_2015.add_prefix("liwc_")

    # Run Foundations LIWC
    mf = analyseLIWC(LIWC_IN, is_foundations=True)
    mf = mf.add_prefix("foundations_")

    merged_df = pd.concat([feature_df, mf, liwc_2015], axis=1, join="inner")
    merged_df.to_csv(feat_type+"_merged.csv", index=False)

    # drop unnecessary cols
    if "post_id" in list(merged_df.columns):
        merged_df.drop(columns=["post_id"])

    return merged_df


def getPrediction(df):

    clf = xgb.XGBRegressor()
    clf.load_model(XGB_PATH)

    y_pred = clf.predict(df)
    return y_pred


@app.route('/', methods=['GET', 'POST'])
def index():
    data = []
    if request.method == 'POST':
        if request.form['submit_posts'] == 'Analyze':

            df_old = getFeatureValues(request.form['old_post'].strip())
            df_old = reorderColumns(df_old)
            y_old = getPrediction(df_old)

            df_new = getFeatureValues(request.form['new_post'].strip())
            df_new = reorderColumns(df_new)
            y_new = getPrediction(df_new)

            changed_list = []
            for i in range(len(df_old)):
                for col in list(df_old.columns):
                    old = df_old.iloc[i][col]
                    new = df_new.iloc[i][col]
                    delta = 0 if old == new else round(new/old-1, 2)

                    changed_list.append({
                        "name": col,
                        "value_old": old,
                        "value_new": new,
                        "perc_change": delta
                    })

            # sort by percentage chagne
            changed_list = sorted(
                changed_list, key=lambda d: d["perc_change"], reverse=True)

            cur_sum = 0
            for x in changed_list:
                if "liwc_" in x["name"]:
                    cur_sum += x["perc_change"]

            cur_sum = 1 if (len(request.form['old_post'].strip()) < 50) or (
                len(request.form['new_post'].strip()) < 50) else cur_sum
            data = [
                {
                    "ahr_old": y_old,
                    "ahr_new": y_new,
                    "model_me": MODEL_ME,
                    "liwc_error": cur_sum == 0,
                    "changedFeatures": changed_list
                }
            ]
        else:
            pass  # unknown
    return render_template('index.html', data=data)


@app.route('/feature_explanation', methods=['GET'])
def feature_explanation():

    if request.method == 'GET':
        feature_name = request.args.get("feature_name")
        feature_explanation = ""

        if "liwc_" in feature_name:
            feature_explanation = "Search for liwc features in the documentation: https://drive.google.com/file/d/1EHrlt6KcL3jZ5gFAA1vjKd5GdXKLRWmk/view?usp=sharing\nSee the list of words associated with the features: https://docs.google.com/spreadsheets/d/16SZS-2UxsvUrEEPvJebmJwYTGJOAEsut/edit?usp=sharing&ouid=100412115316457474517&rtpof=true&sd=true\n"
        elif "foundations_" in feature_name:
            feature_explanation = "See the definition of moral foundations: https://moralfoundations.org/\nSee the entire dictionary: https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic"
        else:
            feature_explanation = FEAT_EXPLANATION[feature_name]
        data = {
            "feature_name": feature_name,
            "feature_explanation": feature_explanation
        }
        return render_template('feature_explanation.html', data=data)


@app.route('/feature_explanation', methods=['GET'])
def shap_analysis():
    if request.method == 'GET':
        return render_template('shap_analysis.html',)


#app.run(host='0.0.0.0', port=3001)
