# We go through the push API to get the ids of every post, then use the official reddit
# API to get the contents of each post and metadata of interest.
import requests
import json
import pandas as pd
import numpy as np
import time
import praw
from praw.models import MoreComments

SHOW_LOGS = True
FIRST_EPOCH = 1370000000 # Right before the first post in 2012
LAST_EPOCH = round(time.time()) 

#TODO: crossposts
# possible pushshift keys:
# ['author', 'author_created_utc', 'author_flair_css_class', 'author_flair_text', 'author_fullname', 'created_utc', 'domain', 'full_link', 'gilded', 'id',
# 'is_self', 'link_flair_css_class', 'link_flair_text', 'media_embed', 'mod_reports', 'num_comments', 'over_18', 'permalink', 'retrieved_on', 'score', 
#'secure_media_embed', 'selftext', 'stickied', 'subreddit', 'subreddit_id', 'thumbnail', 'title', 'url', 'user_reports']
post_columns = {
    "df":        ["post_id", "post_text", "post_title", "post_author_id", "post_score", "post_created_utc", "post_num_comments"], 
    "pushshift": ["id",       "selftext",      "title",         "author",      "score",      "created_utc",      "num_comments"] #Num comments is total amount not only top level
    }
reddit = praw.Reddit(
    client_id="ChMem9TZYJif1A",
    client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
    user_agent="android:com.example.myredditapp:v1.2.3",
)
df_posts = pd.DataFrame(columns=post_columns["df"])
timestamps = list()


def getPostData(post, c):
    values = []
    if SHOW_LOGS:
        print("    GETTING POST "+post["id"]+" ("+str(c)+"/100)")

    for i in post_columns["pushshift"]:
        values.append(post[i])
    arr = np.array([values])
    return arr

def getPushshiftPost(after, before):
    url = 'https://api.pushshift.io/reddit/submission/search/?sort_type=created_utc&sort=asc&subreddit=amitheasshole&after='+ str(after) +"&before"+str(before)+"&size=1000"
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

after = FIRST_EPOCH
while int(after) < LAST_EPOCH:
    try:
        data = getPushshiftPost(after,LAST_EPOCH)
        c = 1
        for p in data:
            post = getPostData(p, c)
            if post.size >0:
                df_posts = df_posts.append(pd.DataFrame(post, columns=df_posts.columns), ignore_index=True)
            c += 1
            
            tmp_time = p['created_utc']
            timestamps.append(tmp_time)
            
        after = timestamps[-1]
        print([str(df_posts.shape[0]) + " posts collected so far."])
        print("----")
        time.sleep(0.1)
    
    except Exception as e:
        print(e)
        df_posts.to_csv("posts_crashed.csv", index=False)
    

# Write to a csv file
df_posts.to_csv("posts.csv", index=False)