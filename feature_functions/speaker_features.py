import logging as lg
from datetime import datetime, timezone

import coloredlogs
import prawcore

coloredlogs.install()

# import global vars
import settings
# TODO: Merge account requests into one call

def get_author_amita_post_activity(account_name):
    """Check how many times account has posted on r/AMITA

    returns : [(str, count)]
        count: number of posts on r/AMITA

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    
    amita_activity_counter = 0
    try:
        author = settings.reddit.redditor(account_name)
        new_submissions = author.submissions.new(limit = None)
        for sub in new_submissions:
            sub_subreddit_name = settings.reddit.submission(id=sub).subreddit.display_name
            amita_activity_counter += sub_subreddit_name == "AmItheAsshole"
            
    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting activity counter to 0\n".format(account_name))

    return [("amita_#posts", amita_activity_counter)]

def get_author_age(account_name):
    """Query post auther age

    returns : [(str, count)]
        count: age in days (rounded down)

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    
    age = 0
    try:
        if account_name != "[deleted]":
            author = settings.reddit.redditor(account_name)
            
            print(author.comment_karma)

            if "created_utc" in vars(author):
                created = author.created_utc
                time_now = datetime.now(tz=timezone.utc)
                time_created = datetime.fromtimestamp(created, tz=timezone.utc)
                age = (time_now-time_created).days
        else:
            lg.warning("\n    Author '{0}' not found. Setting account age to 0\n".format(account_name))
        

    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting account age to 0\n".format(account_name))

    if age > 5879: #Reddit: 23.6.2006
        raise ArithmeticError("Author older than Reddit")
    return [("author_age", age)]

def get_post_author_karma(account_name):
    """Query post auther karma

    returns : [(str, count)]
        count : karma count, 0 if the author has been deleted

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    
    karma = 0
    try:
        if account_name != "[deleted]":
            author = settings.reddit.redditor(account_name)
            karma = author.comment_karma
        else:
            lg.warning("\n    Author '{0}' not found. Setting karma to 0\n".format(account_name))
    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting karma to 0\n".format(account_name))
    return [("author_karma", karma)]
