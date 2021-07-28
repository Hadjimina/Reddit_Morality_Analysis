import prawcore
import logging as lg
from datetime import datetime, timezone
import coloredlogs
coloredlogs.install()

# import global vars
from ..settings import reddit

def get_author_amita_post_activity(account_name):
    """Check how many times account has posted on r/AMITA

    returns count

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    global reddit
    
    amita_activity_counter = 0
    try:
        author = reddit.redditor(account_name)
        new_submissions = author.submissions.new(limit = None)
        for sub in new_submissions:
            sub_subreddit_name = reddit.submission(id=sub).subreddit.display_name
            print(sub_subreddit_name)
            amita_activity_counter += sub_subreddit_name == "AmItheAsshole"
            
    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting activity counter to 0\n".format(account_name))

    return amita_activity_counter

def get_author_age(account_name):
    """Query post auther age

    returns age in days (rounded down)

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    global reddit
    
    age = 0
    try:
        author = reddit.redditor(account_name)
        created = author.created_utc
        time_now = datetime.now(tz=timezone.utc)
        time_created = datetime.fromtimestamp(created, tz=timezone.utc)
        age = (time_now-time_created).days

    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting account age to 0\n".format(account_name))

    if age > 5879: #Reddit: 23.6.2006
        raise ArithmeticError("Author older than Reddit")
    return age

def get_post_author_karma(account_name):
    """Query post auther karma

    returns 0 if the author has been deleted

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    global reddit
    
    karma = 0
    try:
        author = reddit.redditor(account_name)
        karma = author.comment_karma
    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting karma to 0\n".format(account_name))
    return karma