import logging as lg
from datetime import datetime, timezone
import coloredlogs
import prawcore

coloredlogs.install()

# import global vars
import helpers.globals_loader as globals_loader
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

def get_author_info(account_name):
    """Query 
        1. post auther age
        2. post author karma #TODO: Total karma or of r/AMITA? #TODO CHeck obfuscation in PRAW documentation

    returns : [(str, count)]
        count: age in days (rounded down)

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    reddit = globals_loader.reddit
    feature_list = []
    age = 0
    karma = 0
    try:
        if account_name != "[deleted]":
            author = reddit.redditor(account_name)
            print(author.created_utc)
            #Get age
            if "created_utc" in vars(author):
                created = author.created_utc
                time_now = datetime.now(tz=timezone.utc)
                time_created = datetime.fromtimestamp(created, tz=timezone.utc)
                age = (time_now-time_created).days
        
            #Get karma
            if "comment_karma" in vars(author):
                karma = author.comment_karma
        else:
            lg.error("\n    Author '{0}' not found.\n".format(account_name))
        

    except prawcore.exceptions.NotFound:
        lg.error("\n    Exception Author '{0}' not found.\n".format(account_name))
    except AttributeError as error:
        lg.error("\n    Attribute '{0}' not found for {1}.\n".format(error, account_name))

    if age > 5879: #Reddit: 23.6.2006
        lg.warning("Author older than Reddit. Setting to max age")
        age = 5879

    feature_list.append([("author_age", age)])
    feature_list.append([("author_karma", karma)])
    return 



def get_author_age(account_name):
    """Query post auther age

    returns : [(str, count)]
        count: age in days (rounded down)

    Parameters
    ----------
    account_name : str
        reddit account name

    """
    reddit = globals_loader.reddit
    age = 0
    karma = 0
    try:
        if account_name != "[deleted]":
            author = reddit.redditor(account_name)
            

            if "created_utc" in vars(author):
                created = author.created_utc
                time_now = datetime.now(tz=timezone.utc)
                time_created = datetime.fromtimestamp(created, tz=timezone.utc)
                age = (time_now-time_created).days

           
        else:
            #age
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
    print(account_name)
    
    karma = 0
    try:
        if account_name != "[deleted]":
            author = globals_loader.reddit.redditor(account_name,fetch=True)
            print(type(author))
            #Get karma
            #print(author.comment_karma)
            print(vars(author))
            if "comment_karma" in vars(author):
                karma = author.comment_karma
        else:
            lg.warning("\n    Author '{0}' not found. Setting karma to 0\n".format(account_name))
    except prawcore.exceptions.NotFound:
        lg.warning("\n    Author '{0}' not found. Setting karma to 0\n".format(account_name))
    
    return [("author_karma", karma)]
