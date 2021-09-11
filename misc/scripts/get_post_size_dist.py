import pandas as pd
import sys
sys.path.insert(0,'../..')
import constants as CS
import matplotlib.pyplot as plt

#TODO
df = pd.read_csv(CS.POSTS_CLEAN)
post_lengths = df["post_text"].apply(lambda x: len(str(x).split(" "))).to_list()

plt.hist(post_lengths, bins=len(set(post_lengths)))  # density=False would make counts
plt.ylabel('Nr. of posts')
plt.xlabel('Post length [Nr. of words]');
plt.show()