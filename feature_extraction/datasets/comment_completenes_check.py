import pandas as pd

import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import helpers.globals_loader as globals_loader


globals_loader.init()  

df_comments = globals_loader.df_comments
df_posts = globals_loader.df_posts
arr_posts = df_posts["post_id"].to_list()
arr_comments = df_comments["post_id"].to_list()

common = list(set(arr_comments).intersection(arr_posts))
print("We have {0} posts".format(len(arr_posts)))
print("    We have {0} unique posts".format(len(set(arr_posts))))
print("We have {0} comments".format(len(arr_comments)))
print("    We have {0} unique comments".format(len(set(arr_comments))))
print("Intersection of size {0}".format(len(common)))
print("--------------------")
print("This means we have comments for {0}% of our posts".format(round(len(common)*100/len(arr_posts))))

