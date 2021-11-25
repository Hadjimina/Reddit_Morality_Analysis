#from helpers.helper_functions import dict_to_feature_tuples
import re
import coloredlogs
import constants as CS
import statistics as st
import re
from helpers.helper_functions import *
import helpers.globals_loader as globals_loader

from nrclex import NRCLex
from better_profanity import profanity

coloredlogs.install()

def get_punctuation_count(post_text):
    """Count how many times certain punctuation symbols occur in text

    Args:
        post_text (string): Full body text of an r/AITA post

    Returns:
        list: list of tuples e.g. [("!_count", 10), ("?_count", 2), ...]
    """

    symbols = ["!",'"', "?"]
    symbol_dict = dict.fromkeys(symbols, 0)

    #filter out hyperlinks
    cleaned = get_clean_text(post_text,
                    globals_loader.nlp,
                    remove_URL=True,
                    remove_punctuation=False,
                    remove_newline=False,
                    merge_whitespaces=False,
                    do_lowercaseing=False,
                    remove_stopwords=False,
                    do_lemmatization=False)

    post_text = re.sub(r'http\S+', '', cleaned)
    
    for i in range(len(symbol_dict.keys())):
        symbol = list(symbol_dict.keys())[i]
        symbol_dict[symbol] = post_text.count(symbol)

    tuple_list = dict_to_feature_tuples(symbol_dict, "_count")
    return tuple_list


def check_wibta(post_text):
    """ Check if this post is "Am I the asshole..." or "Would I be the asshole..."

    Args:
        post_text (string): Full body text of r/AITA post

    Returns:
       list:  list of tuples e.g. [("is_wita": 0),...]

    """
    expressions_to_check = string_matching_arr_append_ah(CS.WIBTA)
    post_text_clean = get_clean_text(post_text, globals_loader.nlp, do_lemmatization=False, remove_punctuation=True )

    regex = "\\b("+"|".join(expressions_to_check)+")\\b" 
    if re.search(regex, post_text_clean):
        return [("is_wibta", 1)]
    else:
        return [("is_wibta", 0)]



def get_feats_dict(morph_feats):
    """Generate dictionary from feature morphology (i.e. spit out Tense=past|Voice=... => to dict)

    Args:
        morph_feats (string): Morphology features

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
        Emotions: anger, fear, anticipation, trust, surprise, sadness, joy and disgust
        Additionally: positive, negative

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        tuple_list:  e.g. [("joy_norm": 10), ("joy_abs": 0.10)]
    """
    analysed_text = NRCLex(post_text)
    abs_affect = analysed_text.raw_emotion_scores

    # Initialize all emotions that have 0 raw count
    key_set = set(CS.EMOTIONS+list(abs_affect.keys()))    
    for key in key_set:
        if not key in abs_affect:
            abs_affect[key] = 0

    emo_dict = get_abs_and_norm_dict(abs_affect, len(post_text))
    ret_tpl = dict_to_feature_tuples(emo_dict)
    return ret_tpl

def aita_location(post_text):
    """Get the location (in % of entire text) and the number of "aita?" questions the author asks. 
        Get first, last, average location and number of occurences.

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        ret_tuple_list:  [("aita_count": 10), ("aita_avg_location_ratio": 0.10)]
    """    
    # single_words = ["aita", "wita", "wibta"]
    # aita_rgx = "(am i) ((\b(\w*\S\w*)\b) )?(the|a|an) (asshole)" # am i[potentially 1 word] the asshole (e.g. am i really the asshole)

    aita_strings = CS.AITA+CS.WIBTA
    aita_strings = string_matching_arr_append_ah(aita_strings)

    post_text = get_clean_text(post_text, None, remove_punctuation=False, do_lemmatization=False, remove_am=False)

    occurences = 0  
    location_abs = []
    for aita_string in aita_strings:
        cur_occurences = post_text.count(aita_string)
        occurences += cur_occurences
        # if we know we have some occurences get all the start indices
        if cur_occurences > 0:
            location_abs += [m.start() for m in re.finditer(aita_string, post_text)]

    post_length = len(post_text)
    location_ratio = list(map(lambda x: x/post_length, location_abs))
    location_ratio.sort()

    lst = location_ratio[-1] if len(location_ratio) > 0 else -1
    fst = location_ratio[0] if len(location_ratio) > 0 else -1
    mean_loc_ratio =  st.mean(location_ratio) if len(location_ratio) > 0 else -1 
    
    # if we have not found any matches the mean_loc_ratio is -1
    ret_tuple_list =  [("aita_count",occurences), ("aita_avg_location",mean_loc_ratio), ("aita_fst_location", fst), ("aita_lst_location", lst)]
    return ret_tuple_list

def get_sentiment_in_spacy(doc):
    """ We get the sentiment of the spacy doc

     Args:  spaCy doc:  object containing tokenized post text (see https://spacy.io/api/doc)

     Returns: 
        int, int: polairty and subjectivity values
        polarity[-1,1] = sentiment => -1 = negative sentence
        subjectivity [-1,1] => -1 = unsubjective
    """
    return doc._.polarity, doc._.subjectivity

def get_voice_in_spacy(sentence):
    """We get the voice of a specific sentence. 
       If we have active and passive voice in one sentence we check if "nsubj" or "nsubjpass" comes first and consider it as active or passive respecitively.

     Args:  sentence: sentence analysed by spacy

     Return: str: string value of the voice
    """
    filtered = list(filter(lambda x: x.dep_ == "nsubjpass" or x.dep_ == "nsubj", sentence))
    filtered = list(map(lambda x: "passive" if x.dep_ == "nsubjpass" else "active", filtered))
    return filtered[0] if len(filtered) > 0 else ""
    

def get_tense_in_spacy(sentence):
    """We return the tense of a sentence (past, present or future)

    Args:  
        sentence: sentence analysed by spacy

    Return: 
        str : string value of tense
    """
    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py    
    future_flag = False   
    present_flag = False   
    for token in sentence:
        # Sentence is in past if it contains a token with POS (Parts-of-speech) tag equal to  vbd (verb, past tense) or vbn (verb, past participle)
        if token.tag_ in ["VBD", "VBN"]:
            return "past"
        # https://github.com/explosion/spaCy/discussions/2767 
        # VB (verb base form)
        elif future_flag and token.tag_ == "VB":
            return "future"
        
        future_flag = token.dep_ == 'aux' and token.text in ['will', 'shall']
        
        present_flag = present_flag or token.tag_ in ["VBG", "VBP", "VBZ", "VB"] # VBG (verb, gerund or present participle), VBP (verb, non-3rd person singular present), VBZ (verb, 3rd person singular present), VB (verb, base form)
        
    if present_flag:
        return "present"   
        
    return ""


def find_focus_str(pronoun):
    """ Create focus string to determine internal or external focus. It can determine of different types of pronouns
        Returns None if pronoun not found in pronouns list

    Args: 
        pronoun (str): String which we want to map onto the pronouns list

    Returns:
        str (string, None): e.g. "focus_i" or "focus_he" or "focus_you_pl"
    
    """
    pronouns = CS.PRONOUNS

    for lst in pronouns:
        if pronoun in lst:
            focus_str = "focus_"+lst[0]
            lst_idx = pronouns.index(lst)
            focus_str += "_sg" if lst_idx == 1 else ""
            focus_str += "_pl" if lst_idx == 4 else ""
            focus_str = focus_str.replace("rselves", "") #slightly hacky
            return focus_str
    
    return None

def get_profanity_count(post_text):
    """Count how many profane words appear in text and also normalise that value by the number of words

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        tuple list:  e.g. [("profanity_abs": 10), ("profanity_norm", 0.1)]
    """

    words = post_text.split()
    profanity_abs = 0
    for w in words:
        profanity_abs += profanity.contains_profanity(w)

    profanity_norm = profanity_abs / max( len(words), 1) 
    return [("profanity_abs", profanity_abs), ("profanity_norm",profanity_norm)]

def get_focus_in_spacy(token, count_possesive_pronouns=True):
    """ Check if  token in the text is a pronoun. If so, check if they are subjects, objects or possesive pronouns (optional). 
        
    Args:
        token (spaCy token): token we got from analysing the entire post text using spaCy

    Returns:
        str: the focus string with its type (subject, object, possessive) post pended
    """
    token_str = str(token).lower().strip()
    # disregard tokens with only whitespaces or only numbers
    if str(token).isspace() or token_str.isnumeric():
        return None

    # Get focus string
    focus_str = find_focus_str(token_str)
    if focus_str is None:
        return None
    
    if token.dep_ == "nsubj":
        return focus_str+"_subj"
    elif token.dep_ in ["iobj", "dobj", "pobj", "dative"]: # object 
        return focus_str+"_obj"
    elif token.dep_ == "poss" and count_possesive_pronouns:
        return focus_str+"_poss"
    
    return None 

def get_profanity_self_vs_other_in_spacy(sentence):
    """ Iterate over text and
        1. Determine for each sentence if it is about the self (i.e. subject) or about other people 
        2. Sum all raw profanity scores for all sentences. Distinguish between subject I and subject other people
        Returns empty dictionary if subject is not in pronouns list
        
    Args:
        sentence (spaCy doc sentence):  object containing tokenized post text (see https://spacy.io/api/doc)

    Returns:
         ret_dict (dict): dictioanry with profanity values normalised by sentence length. Prefixed with either belonging to "self_" or "other_" focus
    """

    ret_dict = {}
    sent_subjects = [token for token in sentence if (token.dep_ == "nsubj") ] #TODO: should we only check subject?
    profanity_abs = 0

    if len(sent_subjects)>0:
        for w in str(sentence).split():
            profanity_abs += profanity.contains_profanity(w)

    for subj in sent_subjects:
        # break once we have values for self and other
        if any("self" in str for str in list(ret_dict.keys())) and any("other" in str for str in list(ret_dict.keys())):
            break
        
        focus_str = find_focus_str(str(subj))

        if not focus_str is None:
            prefix = "self_" if focus_str == "focus_i" else "other_"

            if any(prefix in str for str in list(ret_dict.keys())):
                continue

            ret_dict[prefix+"prof"] = profanity_abs
            ret_dict[prefix+"prof_length"] = len(sentence.split(" "))

    return ret_dict


def get_emotions_self_vs_other_in_spacy(sentence):
    """ Iterate over text and
        1. Determine for each sentence if it is about the self (i.e. if the subject is a first person pronoun) or about other people (i.e. subject not first person pronoun)
        2. Sum all raw emotion scores for all sentences for each type of emotions up. 
        Returns empty dictionary if subject is not in pronouns list
        
     Args:
        sentence (spaCy doc sentence): object containing tokenized post text (see https://spacy.io/api/doc)

    Returns:
         ret_dict (dictionary): dictionary with raw emotion values. Prefixed with either belonging to "self_" or "other_" focus
    """
    ret_dict = {}
    sent_subjects = [token for token in sentence if (token.dep_ == "nsubj") ] #TODO: should we only check subject?
    analysed_text = NRCLex(str(sentence))
    abs_affect = analysed_text.raw_emotion_scores 
    for subj in sent_subjects:
        # break once we have values for self and other
        if any("self" in str for str in list(ret_dict.keys())) and any("other" in str for str in list(ret_dict.keys())):
            break
        
        focus_str = find_focus_str(str(subj))

        if not focus_str is None:
            prefix = "self_" if focus_str == "focus_i" else "other_"
            if any(prefix in str for str in list(ret_dict.keys())):
                continue

            keys = [prefix+k for k in list(abs_affect.keys())]
            tmp_dict = dict(zip(keys, list(abs_affect.values())))

            # Sum up values
            for k in tmp_dict.keys():
                if k in ret_dict:
                    # TODO: remove this
                    print("ERROR")
                else:
                    ret_dict[k] = tmp_dict[k]

    return ret_dict 


def get_spacy_features(post_text):
    """ Iterate through text and 
        1. Get tense -> get_tense_in_spacy()
        2. Get voice -> get_voice_in_spacy()
        3. Get sentiment -> get_sentiment_in_spacy()
        4. Get internal/external focus -> get_focus_in_spacy()
        5. Get self/oth emotions -> get_emotions_self_vs_other_in_spacy()
        6. Get self/oth profanity -> get_profanity_self_vs_other_in_spacy()
        
        Some use spacy strings listed on https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
       
    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        tuple list ( [(str, int)] ):  e.g. [("future_count": 10), ("future_perc": 0.10),...]
    """
    
    #post_text = "I am going home. I went home. I will go home."
    doc = get_clean_text(post_text, globals_loader.nlp)

    # Dictionary setup
    tenses = ["past", "present", "future"]
    tense_dict = dict.fromkeys(tenses, 0)

    voices = ["active", "passive"]
    voice_dict = dict.fromkeys(voices, 0)

    focus = ["focus_i", "focus_you_sg", "focus_he", "focus_we", "focus_you_pl","focus_they"]
    focus = [x+"_"+y for y in ["subj", "obj", "poss"]for x in focus]
    focus_dict_raw = dict.fromkeys(focus, 0)
    #focus_int_ext = {"internal_focus":0, "external_focus":0}

    # 3. Get Sentiment & Polarity
    sent_dict = {"sent_polarity":0, "sent_subjectivity": 0} 
    if get_sentiment_in_spacy in CS.SPACY_FUNCTIONS:
        pol, subj = get_sentiment_in_spacy(doc)
        sent_dict = {"sent_polarity":pol, "sent_subjectivity": subj} 
        

    # 5. Get emotions in sentence about self vs other    
    emo_self_vs_oth_dict_keys = ["self_"+k for k in CS.EMOTIONS] + ["other_"+k for k in CS.EMOTIONS]
    emo_self_vs_oth_dict = dict.fromkeys(emo_self_vs_oth_dict_keys,0)

    # 6. Get profanity in sentence about self vs other 
    prof_self_vs_oth_dict = {"self_prof":0, "other_prof":0, "self_prof_length":0, "other_prof_length":0 }
    
    for sentence in doc.sents:

        # 1. Get tense
        if get_tense_in_spacy in CS.SPACY_FUNCTIONS:
            tense = get_tense_in_spacy(sentence)
            if tense != "":
                tense_dict[tense] += 1

        # 2. Get voice
        if get_voice_in_spacy in CS.SPACY_FUNCTIONS:
            sentence_voice  = get_voice_in_spacy(sentence)
            if not sentence_voice == "":
                    voice_dict[sentence_voice] += 1
         
        # 5. Get self/other emotions
        if get_emotions_self_vs_other_in_spacy in CS.SPACY_FUNCTIONS:
            tmp_self_oth_emo = get_emotions_self_vs_other_in_spacy(sentence)
            for key in tmp_self_oth_emo.keys():
                emo_self_vs_oth_dict[key] += tmp_self_oth_emo[key]

        # 6. Get self/other profanity
        if get_profanity_self_vs_other_in_spacy in CS.SPACY_FUNCTIONS:
            tmp_self_oth_prof = get_profanity_self_vs_other_in_spacy(sentence)
            for key in tmp_self_oth_prof.keys():
                prof_self_vs_oth_dict[key] += tmp_self_oth_prof[key]
            
        
                
        for token in sentence:
            # 4. Get focus 
            if get_focus_in_spacy in CS.SPACY_FUNCTIONS:
                focus_str = get_focus_in_spacy(token)
                if focus_str in focus_dict_raw.keys():
                    focus_dict_raw[focus_str] += 1


    to_return = []
    nr_sentences = len(list(doc.sents))
    nr_words = len(post_text.split())
    post_length = len(post_text)
    
    # 1. Get tense in tuple list
    tense_dict = get_abs_and_norm_dict(tense_dict, out_off_ratio=nr_sentences)
    to_return += dict_to_feature_tuples(tense_dict)

    # 2. Get voice in tuple list       
    voice_dict = get_abs_and_norm_dict(voice_dict, out_off_ratio=nr_sentences)
    to_return += dict_to_feature_tuples(voice_dict)

    # 3. Get sentiment in tuple list
    to_return += dict_to_feature_tuples(sent_dict)

    # 4. Get focus in tuple list
    # Raw values probably do not make sense since we cannot distinguish between you (singular) and you (plural)
    ## to_return += dict_to_feature_tuples(focus_dict_raw)
    #focus_int_ext["internal_focus"] = focus_dict_raw["focus_i"] # TODO: should we count "focus_we" as internal focus aswell?
    #focus_int_ext["external_focus"] = sum(list(focus_dict_raw.values())) - focus_dict_raw["focus_i"]
    focus_norm_abs = get_abs_and_norm_dict(focus_dict_raw, out_off_ratio=nr_words)
    to_return += dict_to_feature_tuples(focus_norm_abs)
    
    # 5. Get self/other emotions in tuple list
    #emo_self_vs_oth_dict = get_abs_and_norm_dict(emo_self_vs_oth_dict, out_off_ratio=nr_words, only_norm=True)
    # 5.1 Normalise by self focused sentences length and other focused sentence length
    emo_self_vs_oth_dict["self_prof"] = emo_self_vs_oth_dict["self_prof"]/emo_self_vs_oth_dict["self_prof_length"]
    emo_self_vs_oth_dict["other_prof"] = emo_self_vs_oth_dict["other_prof"]/emo_self_vs_oth_dict["other_prof_length"]
    #5.2 remove length entries
    emo_self_vs_oth_dict.pop("self_prof_length", None)
    emo_self_vs_oth_dict.pop("other_prof_length", None)
    to_return += dict_to_feature_tuples(emo_self_vs_oth_dict)

    # 6. Get self/other profanity in tuple list
    prof_self_vs_oth_dict = get_abs_and_norm_dict(prof_self_vs_oth_dict, out_off_ratio=nr_words, only_norm=True)
    to_return += dict_to_feature_tuples(prof_self_vs_oth_dict)

    
    return to_return
    
