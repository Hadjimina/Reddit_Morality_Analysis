import pandas as pd
import numpy as np
import time
import praw
from praw.models import MoreComments
import threading


SHOW_LOGS = True
DF_PATH = "posts_cleaned_27_6_2021.csv"
NUM_THREADS = 1000
# Possible keys found at https://github.com/praw-dev/praw/blob/c818949c848f4520df08b16c098f80a41e897ab5/praw/models/reddit/comment.py
comment_columns = { 
    "df":   ["post_id", "comment_id", "comment_text", "comment_author_id", "comment_score", "comment_created_utc", "comment_was_edited"],
    "praw": [                   "id",         "body",            "author.id",      "score",         "created_utc",             "edited"] #useless atm
    }
reddit = praw.Reddit(
    client_id="ChMem9TZYJif1A",
    client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
    user_agent="android:com.example.myredditapp:v1.2.3",
)

class commentThread (threading.Thread):
    def __init__(self, threadID, post_ids):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.post_ids = post_ids
      self.df = pd.DataFrame(columns=comment_columns["df"])

    def getCommentsData(_, id):
        submission = reddit.submission(id=id)
        values = []
        counter = 1
        for top_level_comment in submission.comments:
            #if SHOW_LOGS:
            #    print("    GETTING COMMENT "+top_level_comment.id+ " ("+str(counter)+"/"+str(len(submission.comments))+")")

            if isinstance(top_level_comment, MoreComments):
                continue

            value_i = []
            value_i.append(id)
            value_i.append(top_level_comment.id)
            value_i.append(top_level_comment.body)
            if top_level_comment.author and hasattr(top_level_comment.author, 'id'):
                value_i.append(top_level_comment.author.id)
            else:
                value_i.append(None)
            value_i.append(top_level_comment.score)
            value_i.append(top_level_comment.created_utc)
            value_i.append(top_level_comment.edited)

            values.append(value_i)
            counter+=1
        return np.array(values)

    def run(self):
        counter_thread = 0  
        for j in self.post_ids:
            counter_thread +=1
            if SHOW_LOGS:
                length = self.post_ids.shape[0]
                timestamp = time.strftime('%H:%M:%S')
                print("THREAD "+str(self.threadID)+"/"+str(NUM_THREADS)+ " Row "+str(counter_thread)+"/"+str(length)+" "+str(timestamp))
            comments = self.getCommentsData(j)
            if comments.size > 0:
                        self.df = self.df.append(pd.DataFrame(comments, columns=self.df.columns), ignore_index=True)




        
df_comments = pd.DataFrame(columns=comment_columns["df"])
df_posts = pd.read_csv(DF_PATH) 
post_ids = df_posts["post_id"]
spacing = np.linspace(0, post_ids.shape[0]-1, NUM_THREADS, endpoint=True, dtype=int)

threadLock = threading.Lock()
threads = list()

for i in range(len(spacing)):
    if i == range(len(spacing))[-1]:
        break
    threads.append(commentThread(i, post_ids.loc[ spacing[i] : spacing[i+1]-1 ]))    

for t in threads:
    t.start()

for t in threads:
    t.join()

for t in threads:
    df_comments = pd.concat([df_comments, t.df])

df_comments.to_csv("comments.csv", index=False)