import logging as lg
from datetime import datetime, timezone
import coloredlogs
import prawcore
import re
import constants as CS

coloredlogs.install()

# import global vars
import helpers.globals_loader as globals_loader
from helpers.helper_functions import *

def get_author_info(account_name):
    """ Get information about the post author. Namely get the author account age and account karma (link & comment)

    Args:
        account_name: name of account for post

    Returns:
        feature_list: list of features tuples e.g. [("account_age", age), ("account_comment_karma", comment_karma)]
    """ 
    reddit = globals_loader.reddit


    feature_list = []
    age = 0
    comment_karma = 0
    link_karma = 0
    try:
        if account_name != "[deleted]":
            author = reddit.redditor(account_name)
            print(author.created_utc) #THIS PRINT IS REQUIRED TO MAKE author OBJECT NON-LAZY
            #Get age
            if "created_utc" in vars(author):
                created = author.created_utc
                time_now = datetime.now(tz=timezone.utc)
                time_created = datetime.fromtimestamp(created, tz=timezone.utc)
                age = (time_now-time_created).days
        
            #Get comment karma
            if "comment_karma" in vars(author):
                comment_karma = author.comment_karma

            #Get comment karma
            if "link_karma" in vars(author):
                link_karma = author.link_karma
        else:
            lg.error("\n    Author '{0}' not found.\n".format(account_name))

    except prawcore.exceptions.NotFound:
        lg.error("\n    Exception Author '{0}' not found.\n".format(account_name))
    except AttributeError as error:
        lg.error("\n    Attribute '{0}' not found for {1}.\n".format(error, account_name))

    if age > 5879: #Reddit: 23.6.2006
        lg.warning("Author older than Reddit. Setting to max age")
        age = 5879

    feature_list += [("account_age", age)]
    feature_list +=[("account_comment_karma", comment_karma)]
    feature_list +=[("account_link_karma", link_karma)]
    return feature_list


def get_author_age_and_gender(post_text):
    """ Extract age and gender from post text e.g. "I (24 M)" for post author

        Args:
            post_text (string):  full body of post on AITA

        Returns:
           [(str, int)]:  e.g. [("author_age": 10), ("author_gender": 1),...] 

    """

    cleaned_text = prep_text_for_string_matching(post_text) 

    # extract all ages
    rgx_age_gender = re.compile(r"(\d{1,2}\s?(f|m|female|male)[^a-zA-Z0-9]{1})|([^a-zA-Z0-9]{1}(f|m|female|male)\s?\d{1,2})")
    all_ages_genders_w_span = []
    for match in rgx_age_gender.finditer(cleaned_text):
        match_span = list(match.span())
        match_str = match.group().replace(" ", "")
        if match.group(0)[0] == " ":
            match_span[0] = match_span[0]+1
        if match.group(0)[-1] == " ":
            match_span[1] = match_span[1]-1
        match_span = (match_span[0], match_span[1])
        all_ages_genders_w_span.append([match_span, match_str])
    
    # create age, gender tuple list
    age_gender_tpl = []
    for span, age_gend_str in all_ages_genders_w_span:
        age = []
        gender = []
        for c in age_gend_str:
            if c.isdigit():
                age.append(c)
            elif c.isalpha():
                gender.append(c)
        age_int = int("".join(age))
        if len(age)>0 and len(gender)>0:
            age_gender_tpl.append([span,(age_int, gender[0])])
    
    if not len(age_gender_tpl) > 0:
        return [("author_age",-1), ("author_gender", -1)]
    #print(age_gender_tpl)
    #print(cleaned_text)

    # extract all pronouns
    cleaned_text = re.sub('[^a-zA-Z0-9 \n\.]', " ", cleaned_text)

    pronouns_flat = [e for sub in CS.PRONOUNS for e in sub]
    pronouns_rgx_str = "|".join(pronouns_flat) #structure needs to be: 1 non alpha numeric char, pronoun 1 non alphnumeric char
    pronouns_rgx_str = "[^a-zA-Z0-9]{1}("+pronouns_rgx_str+")[^a-zA-Z0-9]{1}" 
    rgx_pronoun = re.compile(r""+pronouns_rgx_str)
    pronouns_list = []
    pronoun_keys = set({})
    for match in rgx_pronoun.finditer(cleaned_text):
        match_str = ""
        for lst in CS.PRONOUNS:
            match_group_str = match.group(0)
            match_group_str = match_group_str.strip()
            if match_group_str in lst:
                match_str = lst[0]
        if len(match_str) != 0:
            pronoun_keys.add(match_str)
            pronouns_list.append([match.span(), match_str])

    pronoun_age_gender_dict = {}#dict.fromkeys(list(pronoun_keys), set())

    for span_age_gender, age_gender in age_gender_tpl:
        for span_pronoun, pronoun in pronouns_list:
            sub_str = cleaned_text[span_pronoun[1]:span_age_gender[0]]
            if len(sub_str) < CS.PRONOUN_AGE_GENDER_DIST and len(sub_str) > 0:
                if not pronoun in pronoun_age_gender_dict:
                    pronoun_age_gender_dict[pronoun] = [age_gender]
                elif not age_gender in pronoun_age_gender_dict[pronoun]:
                    pronoun_age_gender_dict[pronoun].append(age_gender)
                #print(pronoun_age_gender_dict)
                #print("---")
                continue
    
    # TODO: should we only look at "i" pronoun or also others?
    author_age = -1
    author_gender = -1
    if "i" in pronoun_age_gender_dict:
        author_gender = int(pronoun_age_gender_dict["i"][0][1] == "f") # author_gender = 1 => author is a woman
        author_age = pronoun_age_gender_dict["i"][0][0]

    return [("author_age",author_age), ("author_gender", author_gender)]



"""

def get_author_amita_post_activity(account_name):
    Check how many times account has posted on r/AMITA

    returns : [(str, count)]
        count: number of posts on r/AMITA

    Parameters
    ----------
    account_name : str
        reddit account name

    reddit = globals_loader.reddit
    amita_activity_counter = 0
    try:
        author = reddit.redditor(account_name)
        new_submissions = author.submissions.new(limit = None)
        for sub in new_submissions:
            sub_subreddit_name = reddit.submission(id=sub).subreddit.display_name
            amita_activity_counter += sub_subreddit_name == "AmItheAsshole"
            
    except prawcore.exceptions.NotFound:
        amita_activity_counter
        #lg.warning("\n    Author '{0}' not found. Setting activity counter to 0\n".format(account_name))
    except prawcore.exceptions.Forbidden:
        amita_activity_counter
        #lg.warning("\n    Author '{0}' forbidden. Setting activity counter to 0\n".format(account_name))

    return [("amita_#posts", amita_activity_counter)]



"""