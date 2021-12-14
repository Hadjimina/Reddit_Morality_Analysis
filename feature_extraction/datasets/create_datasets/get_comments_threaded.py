import pandas as pd
import numpy as np
import time
import praw
from praw.models import MoreComments
import threading
import datetime
from os import listdir
from os.path import isfile, join
import os.path


SHOW_LOGS = True
DF_PATH = "posts_cleaned_27_6_2021.csv"
NUM_THREADS = 100+1
MAX_RETRIES = 10
BACKOFF = 60  #[s]
TMP_FOLDER ="tmp_comments/"
TMP_MERGED_FILENAME = "tmp_merged.csv"
DO_RECOVERY = True

# Possible keys found at https://github.com/praw-dev/praw/blob/c818949c848f4520df08b16c098f80a41e897ab5/praw/models/reddit/comment.py
comment_columns = { 
    "df":   ["post_id", "comment_id", "comment_text", "comment_author_id", "comment_score", "comment_created_utc", "comment_was_edited"],
    "praw": [                   "id",         "body",            "author.id",      "score",         "created_utc",             "edited"] #useless atm
    }


class commentThread (threading.Thread):
    def __init__(self, threadID, post_ids):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.post_ids = post_ids
      self.csv_path = TMP_FOLDER+"thread_"+str(threadID)+"_tmp.csv"
      if os.path.exists(self.csv_path):
          self.df = pd.read_csv(self.csv_path)
      else: 
        self.df = pd.DataFrame(columns=comment_columns["df"])
            
    def getCommentsData(_, id):
        submission = praw.Reddit(
                        client_id="ChMem9TZYJif1A",
                        client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
                        user_agent="android:com.example.myredditapp:v1.2.3",
                    ).submission(id=id)

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
                string_to_print = "THREAD "+str(self.threadID)+"/"+str(NUM_THREADS-2)+ " Row "+str(counter_thread)+"/"+str(length)+" "+str(timestamp)
                print(string_to_print)

            for _ in range(MAX_RETRIES):
                try:
                    comments = self.getCommentsData(j)
                    if comments.size > 0:
                        to_append = pd.DataFrame(comments, columns=self.df.columns)
                        self.df = pd.concat([self.df,to_append])
                        self.df.to_csv(self.csv_path, index=False)
                    else:
                        nans = np.array(["EMPTY"*self.df.columns.shape[0]])
                        nans[0] = j
                        to_append = pd.DataFrame(nans, columns=self.df.columns)
                    break
                except TimeoutError:
                    time.sleep(BACKOFF)
                    pass
                #except Exception as e:
                #    print(e)
                #    print("Unexpected error:", sys.exc_info()[0])

        print("Thread "+str(self.threadID)+" done!")


df_comments = pd.DataFrame(columns=comment_columns["df"])
df_posts = pd.read_csv(DF_PATH)

#merge tmp files
if DO_RECOVERY:
    print("DOING RECOVERY")
    tmp_files = [f for f in listdir(TMP_FOLDER) if isfile(join(TMP_FOLDER, f))]
    merged_tmp_dfs = pd.DataFrame(columns=comment_columns["df"])
    for f in tmp_files:
        print(f)
        df_i = pd.read_csv(TMP_FOLDER+f, eninge="c", error_bad_lines=False)
        merged_tmp_dfs = merged_tmp_dfs.append(df_i, ignore_index = True)
    merged_tmp_dfs.to_csv(str(time.time())+"_"+TMP_MERGED_FILENAME, index=False)
    print("  RECOVERY COMMENTS MERGED AND SAVED")
    already_checked_ids = merged_tmp_dfs["post_id"].unique()
    size_old = df_posts.shape[0]
    df_posts = df_posts[~df_posts['post_id'].isin(already_checked_ids)]
    size_new = df_posts.shape[0]
    removed = size_old-size_new
    print("  REMOVED "+str(removed)+"/"+str(already_checked_ids.shape[0])+" IDS")

start_time = time.time()
post_ids = df_posts["post_id"]
spacing = np.linspace(0, post_ids.shape[0]-1, NUM_THREADS, endpoint=True, dtype=int)

threadLock = threading.Lock()
threads = list()

print("STARTING COMMENTS QUERY")
print("#THREADS: "+str(NUM_THREADS))
print("CSV: "+str(DF_PATH))
print("TOTAL POSTS: "+str(df_posts.shape[0]))

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

d1 = datetime.date.today().strftime("%d_%m_%YYYY")
df_comments.to_csv("comments_"+d1+".csv", index=False)
print("Average execution time: "+str((time.time() - start_time)/df_posts.shape[0]))