"""
helper_functions.py
"""
import json
import string
from datetime import datetime, date
import re
import os
import sys
import requests
import nltk
import socket
import emoji
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
#import helpers.globals_loader as globals_loader

# alternatively can be done using spacy? gensim?


def get_clean_text(post_text,
                   nlp,
                   remove_URL=True,
                   remove_punctuation=0,
                   remove_newline=True,
                   merge_whitespaces=True,
                   do_lowercaseing=True,
                   remove_stopwords=False,
                   do_lemmatization=True,
                   remove_am=False):
    """Function to clean text (i.e. remove urls, punctuation, newlines etc) and do lemmatizaion if needed

    Args:
        post_text (string): full body text of r/AITA posts
        nlp (function): nlp function from the spacy object
        remove_URL (bool, optional): Whether or not URLs should be removed. Defaults to True.
        remove_punctuation (int, optional): Whether or not punctuation should be removed. Defaults to False.
        remove_newline (bool, optional): Whether or not newline characters should be removed. Defaults to True.
        merge_whitespaces (bool, optional): Whether or not mulitple consecutive whitespace should be merged to one. Defaults to True.
        do_lowercaseing (bool, optional): Whether or not text should be lowercased. Defaults to True.
        remove_stopwords (bool, optional): Whether or not stopwords from nltk should be removed (includes here, than, myself, which, it....). Defaults to False.
        do_lemmatization (bool, optional): Whether or not we should return the lemmatized post. Defaults to True.
        remove_am (bool, optional): Whether or not we should remove all "'m" and "am" in the post. Defaults to False

    Returns:
        string/spacy doc: Cleaned string or cleaned & lemmatized spacy doc. Spacydoc can be iterated over just like one would a string
    """

    #remove emojis:
    post_text = emoji.get_emoji_regexp().sub(u'', post_text)

    if remove_am:
        post_text = post_text.replace("'m", " ").replace("am", " ")

    if remove_URL:
        post_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(post_text))

    if remove_punctuation == 1:
        post_text = post_text.translate(
            str.maketrans(' ', ' ', string.punctuation))
    elif remove_punctuation == 2:
        post_text = post_text.translate(str.maketrans(
            string.punctuation, ' '*len(string.punctuation)))

    # \n = newline & \r = carriage return
    if remove_newline:
        post_text = post_text.replace('\n', ' ').replace('\r', '')

    if merge_whitespaces:
        post_text = ' '.join(post_text.split())

    if do_lowercaseing:
        post_text = post_text.lower()

    # removes things like [i, me, my, myself, we, our, ours, ...
    if remove_stopwords:
        post_text = " ".join(
            [word for word in post_text.split() if word not in stopwords.words('english')])

    if do_lemmatization:
        return nlp(post_text)  # spacy
    else:
        return post_text

def get_ups_downs_from_ratio_score(r, s):
    """Given a ratio and a score, we will return individual number of upvotes and downvotes & ratio, score
        We have to do this since, reddit remove the possibility to check the exact amount of upvotes and downvotes a few years ago. 
        Also the ratio is rounded so we will not get completely accurate values.

        We get formula since we know that x-y=s and x/x+y = r

    Args:
        r (float): upvote ratio from reddit
        s (int): the current score of the post/comment

    Returns:
        list: list with first element being the number of upvotes and the second one the number of downvotes, followed by the upvote ratio and current score
    """
    if r is None or s is None:
        print("NONE")
        return [None, None, None, None]

    # if we have have a 50% upvote ratio we set the upvotes to 0
    ups = round((r * s) / (2 * r - 1)) if r != 0.5 else 0
    downs = round(ups - s)
    return [ups, downs, r, s]


def dict_to_feature_tuples(dict, suffix=""):
    """ Take a dict at end of a feature function and converts it into the tuple format suitable for the dataframe

    Args:
        dict: dictionary containing features data
        suffix: string we post pend to each feature category

    Returns:
        tuple_list: list of tuples for the dataframe e.g. [(feature name, value),...]
    """
    tuple_list = []
    #all_values_zero = True
    for k, v in dict.items():
        tpl = (("{0}"+suffix).format(k), v)
        # if not np.isclose(v, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        #    all_values_zero = False
        tuple_list.append(tpl)

    return tuple_list  # if not all_values_zero else []


def prep_text_for_string_matching(text):
    """Prepare text for string matching in two steps
        1. make all text lowercase
        2. replace multiple whitespace with only one whitespace
    Args:
        text (str): some text we want to perform string matching on

    Returns:
        str: text prepared for string matching
    """
    print("THIS FUNCTON IS DEPRECATED USE get_clean_text INSTEAD")
    text = text.lower()

    text = " ".join(text.split())
    return text


def string_matching_arr_append_ah(matches):
    """When performing string matching we sometimes manualy specific which words to match. Often these include the words "asshole".
        Often users on AITA, do not write "asshole" but write "ah" instead. Thus we need to extend the matching list but replace all "asshole" occurences with "ah"

    Args:
        matches (list): list of matching strings, these do not include any "ah" yet but only "asshole"

    Returns:
        matches_extended: list of matching strings extended to include "asshole" and "ah
    """

    asshole_str = "asshole"
    ah_str_list = ["ah","a-hole", "a hole", "ass hole", "a**hole", "a** hole"]

    ah_to_post_pend = []
    for ah_str in ah_str_list:
        for match_str in matches:
            if asshole_str in match_str:
                ah_to_post_pend += [match_str.replace(asshole_str, ah_str)]

    matches_extended = matches + ah_to_post_pend
    return matches_extended


def get_abs_and_norm_dict(abs_dict, out_off_ratio, append_abs=True, only_norm=False):
    """Get a feature dictinoary containing only absolute values and extended it to include the normalised values aswell.

    Args:
        abs_dict (dict): dict of calucluated features with only absolute features
        append_abs (bool, optional): whether or not we should append the string "_abs" to the exisitng keys in abs_dict
        only_norm (bool, optional): whether or not we should return only the normalised values

    Returns:
        complete_dict: dict of calucluated features with absolute and normalised features
    """
    features = list(abs_dict.keys())
    # create abs and perc values
    abs_postpend = "_abs" if append_abs else ""
    all_keys = [x+"_norm" for x in features]
    if not only_norm:
        all_keys = [x+abs_postpend for x in features] + all_keys
    complete_dict = dict.fromkeys(all_keys, 0)

    for v in features:
        curr_value = abs_dict[v]
        if not only_norm:
            complete_dict[v+abs_postpend] = curr_value
        complete_dict[v+"_norm"] = curr_value/max(out_off_ratio, 1)
    return complete_dict


def get_date_str(minified=False):
    """ Get current date in format dd_mm_YYYY or dd_mm_YYYY

    Args:
        minified (bool, optional): whether we only want day and month
    Returns:
        date_time: date string
    """
    now = datetime.now()
    date_time = now.strftime(
        "%d_%m_%Y") if not minified else now.strftime("%d_%m")
    return date_time


def contains_letters_numbers(str):
    """Check if the given string contains letters and numbers. Must contain both to return True. 

    Args:
        str (string): string we want to check

    Returns:
        bool: Whether or not argument contains letters AND numbers
    """

    flag_digit = False
    flag_letter = False
    for i in str:
        flag_digit = flag_digit | i.isdigit()
        flag_letter = flag_letter | i.isalpha()

    return flag_digit & flag_letter


def output_dir_name():
    """Return the name of the output directories parent folder
    """

    today = date.today()
    ouput_folder = today.strftime("%d_%m_%Y")
    return ouput_folder


def flatten_list(lst):
    """Flattens a list to only 1 dimension

    Args:
        lst (list): Any type of list    

    Returns:
        flattened list: Flat list of the original list
    """
    return [item for sublist in lst for item in sublist]


def find_all(a_str, sub):
    """Find all substring occurences (non overlapping)

    Args:
        a_str (string): some string
        sub (string): some substring we want to find within a_str

    Yields:
        list: list of indices
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def sent_telegram_notification(msg):
    """Send a telegram notification for progress updates.

    Args:
        msg (string): message we would like to send
    """
    path = sys.path[0]+"/secrets/telegram.json"
    file = open(path)
    data = json.load(file)
    url = "https://api.telegram.org/bot{secret}/sendMessage".format(
        secret=data["secret"])
    data = {"chat_id": data["chat_id"], "text": msg}
    requests.post(url, data)


def dist_conf_to_function_list_str(dist_conf):
    """Given the distribution configuration object, generate a list of all functions executed on the current host

    Args:
        dist_conf (dict): distriubtion configuration object, see distributed_configuration.py

    Returns:
        function_list_str (string): String list of which feature function were exectuted
    """

    function_list_str = ""
    hostname = socket.gethostname().lower()

    def ff_to_string(x): return "  "+str(x.__name__)+"\n"

    # Feature functions standard
    processing_types = ["mp", "mono"]
    for ptype in processing_types:
        if ptype in dist_conf.feature_functions["hosts"][hostname]:
            for k in dist_conf.feature_functions["hosts"][hostname][ptype].keys():
                ff_list = dist_conf.feature_functions["hosts"][hostname][ptype][k]
                for ff in ff_list:
                    function_list_str += ff_to_string(ff[0])

    # spacy
    if "spacy" in dist_conf.feature_functions["hosts"][hostname]:
        spacy_list = dist_conf.feature_functions["hosts"][hostname]["spacy"]
        for ff in spacy_list:
            function_list_str += ff_to_string(ff)

    # topic modelling, foundations, liwc
    binary_features_list = ["topic", "foundations", "liwc" ]
    for feat in binary_features_list:
        if feat in dist_conf.feature_functions["hosts"][hostname] and dist_conf.feature_functions["hosts"][hostname][feat]:
            function_list_str += "  {fn}\n".format(fn=feat+"_modelling"if feat == "topic" else feat)
        

    return function_list_str

def downsample_to_proportion(rows, proportion):
    """Downsample a list

    Args:
        rows (list): List of rows we want to downsample
        proportion (int): ratio to downsample rows to. e.g. 0.3

    Returns:
        list: downsampleded list
    """
    
    return rows[::int(1 / proportion)]

def find_all(a_str, sub):
    """Find all indices of a substring (sub) in the string a_str

    Args:
        a_str (str): long string we want to find subs indeces in
        sub (str): sub string we get indices of

    Yields:
        [int]: list of sub indices start locations
    """
    
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches
