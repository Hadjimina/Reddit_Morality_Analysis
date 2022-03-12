from helpers.helper_functions import *
import helpers.globals_loader as globals_loader
import logging as lg
from datetime import datetime, timezone
import coloredlogs
import prawcore
import re
import constants as CS

coloredlogs.install()

# import global vars


def get_author_info(account_name):
    """ Get information about the post author. Namely get the author account age and account karma (link & comment)

    Args:
        account_name (string): name of account for post

    Returns:
        feature_list: list of features tuples e.g. [("account_age", age), ("account_comment_karma", comment_karma)]
    """
    # Here we are not able to use .info to aggregate
    reddit = globals_loader.reddit

    feature_list = []
    age = 0
    comment_karma = 0
    link_karma = 0
    try:
        if account_name != "[deleted]":
            author = reddit.redditor(account_name)
            # THIS PRINT IS REQUIRED TO MAKE author OBJECT NON-LAZY
            print(author.created_utc)
            # Get age
            if "created_utc" in vars(author):
                created = author.created_utc
                time_now = datetime.now(tz=timezone.utc)
                time_created = datetime.fromtimestamp(created, tz=timezone.utc)
                age = (time_now-time_created).days

            # Get comment karma
            if "comment_karma" in vars(author):
                comment_karma = author.comment_karma

            # Get comment karma
            if "link_karma" in vars(author):
                link_karma = author.link_karma
        else:
            lg.error("\n    Author '{0}' not found.\n".format(account_name))

    except prawcore.exceptions.NotFound:
        lg.error(
            "\n    Exception Author '{0}' not found.\n".format(account_name))
    except AttributeError as error:
        lg.error("\n    Attribute '{0}' not found for {1}.\n".format(
            error, account_name))

    if age > 5879:  # Reddit: 23.6.2006
        lg.warning("Author older than Reddit. Setting to max age")
        age = 5879

    print(age, comment_karma, link_karma)
    feature_list += [("account_age", age)]
    feature_list += [("account_comment_karma", comment_karma)]
    feature_list += [("account_link_karma", link_karma)]
    return feature_list


def get_author_age_and_gender(post_text):
    """Extract age and gender from post text e.g. "I (24 M)" for pos author

    Args:
        post_text (string): full body of post on AITA

    Returns:
        list: tuple list e.g. e.g. [("author_age": 10), ("author_gender": 1),...] 
    """
    cleaned_text = get_clean_text(
        post_text, None, remove_punctuation=False, do_lemmatization=False, remove_am=True)
    #cleaned_text = prep_text_for_string_matching(post_text)

    # extract all ages
    male_strings = ["male", "m"]
    female_strings = ["female", "f"]
    rgx_pronoun = "\\b(" + \
        "|".join(CS.PRONOUNS[0]+CS.PRONOUN_MATCHING_MISC)+")\\b"
    rgx_gap = "[^b-z0-9]{0,3}"
    rgx_age = "(\\d{1,2})"
    rgx_gender = "("+"|".join(male_strings+female_strings)+")"
    rgx_string = (
        "("+rgx_pronoun+rgx_gap+rgx_age+rgx_gap+rgx_gender+")|" +  # i (23, f)
        "("+rgx_pronoun+rgx_gap+rgx_gender+rgx_gap+rgx_age+")|" +  # i (f, 23)
        "("+rgx_gender+rgx_gap+rgx_age+rgx_gap+rgx_pronoun+")|" +  # (f, 23) i
        "("+rgx_gender+rgx_gap+rgx_pronoun+rgx_gap+rgx_age+")|" +  # (23, f) i
        "("+rgx_pronoun+rgx_gap+rgx_age+")|" +  # i 23
        "("+rgx_pronoun+rgx_gap+rgx_gender+")"  # i f
    )

    age_gender_list = []
    
    rgx = re.compile(r""+rgx_string)

    for match in rgx.findall(cleaned_text):
        groups = list(
            filter(lambda x: (not contains_letters_numbers(x)) and len(str(x)) > 0, match))
        # split matches from whole list into chunks of 3 (b.c. we have 3 group for each regex "line")
        groups_chunked = [groups[i:i+3] for i in range(0, len(groups), 3)]
        for chunk in groups_chunked:
            if len("".join(chunk)) == 0:
                continue
            age = -1
            gender = -1
            for g in chunk:
                if g.isnumeric():
                    age = int(g)
                if g in female_strings:
                    gender = 1
                if g in male_strings:
                    gender = 0
            if age != -1 or gender != -1:
                age_gender_list.append((age, gender))

    # sort list
    both_valid = list(filter(lambda x: x[0]>= 0 and x[1] >= 0, age_gender_list))
    age_gender_list = both_valid if len(both_valid)>0 else age_gender_list
    
    print("AGE GENDER:")
    print(age_gender_list)
    print("--------")
    if len(age_gender_list) > 1:
        True
        #lg.warning("More than 1 age/gender found for poster.")
    elif len(age_gender_list) < 1:
        age_gender_list = [(-1, -1)]  # -1 if nothing found

    return [("author_age", age_gender_list[0][0]), ("author_gender", age_gender_list[0][1])]


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
