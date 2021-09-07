from helpers.helper_functions import dict_to_feature_tuples
import logging as lg
import re
import coloredlogs
import constants as CS

import helpers.globals_loader as globals_loader

from helpers import *
from nrclex import NRCLex


coloredlogs.install()


def get_punctuation_count(post_text,**kwargs):
    """Count how many times certain punctuations syombols occur in text

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        [(str, int)]:  e.g. ("!_count": 10)
    """    

    symbols = ["!",'"', "?"]
    symbol_dict = dict.fromkeys(symbols, 0)

    #filter out hyperlinks
    #print(post_text)
    post_text = re.sub(r'http\S+', '', post_text)

    
    for i in range(len(symbol_dict.keys())):
        
        symbol = list(symbol_dict.keys())[i]
        symbol_dict[symbol] = post_text.count(symbol)

    tuple_list = dict_to_feature_tuples(symbol_dict, "_count")
    #tuple_list = [("{0}_count".format(k), v) for k, v in symbol_dict.items()]
    return tuple_list

def get_feats_dict(morph_feats):
    """Generate dictionary from feature morphology (i.e. spit out Tense=past|Voice=... => to dict)

    Args:
        morph_feats (str): Morphology features

    Returns:
        dict: dict with all features as key and corresponding values
    """

    pairs_split = morph_feats.split("|")
    k = []
    v = []
    for pair in pairs_split:
        kv_split = pair.split("=")
        k.append(kv_split[0])
        v.append(kv_split[1])
    
    feats_dict = dict(zip(k, v)) if len(k) > 0 else {}
    return feats_dict

def get_emotions(post_text):
    """Analyse emotions contained within a text using NRC Word-Emotion Association Lexicon (aka EmoLex)
       Frequencies => ratio to total number of words

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
         [(str, int)]:  e.g. [("joy_freq": 10), ("joy_abs": 0.10)]
    """

    analysed_text = NRCLex(post_text)
    abs_affect = analysed_text.raw_emotion_scores
    freq_affect = analysed_text.affect_frequencies

    # Initialize all emotions that have 0 raw count
    for key in list(freq_affect.keys()):
        if not key in abs_affect:
            abs_affect[key] = 0

    abs_list = dict_to_feature_tuples(abs_affect, suffix="_abs")
    freq_list = dict_to_feature_tuples(freq_affect, suffix="_freq")

    ret_list = abs_list+freq_list
    return ret_list

def get_tense_voice_sentiment(post_text):
    """Iterate through text and 
       1. count how many sentances are in past, present, future tense  
       2. count how many sentances are in active/passive voice
       Above values are returned in absolute number and in percentage of #verbs / #all sentances (Where verbs are defined as "VERBS" from POS tagger)

       3. check if post_text sentiment is postitive or negative => only checked on per text level (not individual sentances)
       
    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
         [(str, int)]:  e.g. [("future_count": 10), ("future_perc": 0.10)]
    """
    doc = globals_loader.nlp(post_text)    

    tenses = ["past", "present", "future"]
    tense_dict = dict.fromkeys(tenses, 0)

    voices = ["active", "passive"]
    voice_dict = dict.fromkeys(voices, 0)

    # Sentiment & Polarity done below
    # polarity[-1,1] = sentiment => -1 = negative sentance, subjectivity [-1,1] => -1 = unsubjective
    sent_dict = {"sent_polarity":doc._.polarity, "sent_subjectivity": doc._.subjectivity } 

    # Profatnity done below
    #prof_dict = {"profanity_abs":0 }
    #token_count = 0

    verb_count = 0
    
    for sentence in doc.sents:
        #print("--NEW SENT--")
        #verb_count_old = verb_count
        sentence_voice = ""
        sentence_tense = ""

        voice_flag = False
        for token in sentence:
            #token_count +=1
            #if token._.is_profane:
            #    prof_dict["profanity"] += 1

            # For tense and voice we only look at verbs
            if token.pos_ == CS.SP_VERB:
                feat_dict = get_feats_dict(str(token.morph))
                #if "Voice" in feat_dict.keys():
                #print(feat_dict)
                # Do tense
                if CS.SP_FEATS_TENSE in feat_dict:
                    verb_count +=1

                    tense = feat_dict[CS.SP_FEATS_TENSE]
                    if tense == CS.SP_TENSE_PAST:
                        sentence_tense = "past"
                    elif tense == CS.SP_TENSE_PRESENT:
                        sentence_tense = "present"
                    elif tense == CS.SP_TENSE_FUTURE:
                        sentence_tense = "future"
                    tense_dict[sentence_tense] += 1


            # Do voice
            # TODO: This is definetly not perfect. Naive implementation only https://stackoverflow.com/questions/19495967/getting-additional-information-active-passive-tenses-from-a-tagger
            if voice_flag:
                continue

            if "nsubjpass" == token.dep_:
                sentence_voice = "passive"
            elif "nsubj" == token.dep_:
                sentence_voice = "active"

            if not sentence_voice == "":
                voice_flag = True
                voice_dict[sentence_voice] += 1

    # Calculate absolute profanity
    #prof_dict["profanity_perc"] = prof_dict["profanity_abs"]/max(token_count,1)
        
                            
    # Merge & get percentage values for tense and voice
    merged_dict = {**tense_dict, **voice_dict,}

    mrg_keys = [s + "_perc" for s in merged_dict.keys()]
    
    mrg_values = list(map(lambda x: x/max(verb_count,1), merged_dict.values()))
    perc_dict = dict(zip(mrg_keys, mrg_values))

    
    abs_features = dict_to_feature_tuples(merged_dict, suffix= "_abs")
    perc_features = dict_to_feature_tuples(perc_dict)
    sent_features = dict_to_feature_tuples(sent_dict)

    #prof_features = dict_to_feature_tuples(prof_dict)
    #print(prof_features)

    feat_list = abs_features+perc_features+sent_features#+prof_features
    
    return feat_list
    
    