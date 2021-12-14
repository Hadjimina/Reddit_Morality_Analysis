import pandas as pd
import sys
sys.path.insert(0,'../..')
import constants as CS

df = pd.read_csv(CS.COMMENTS_CLEAN)
result = pd.DataFrame([[]])
avg_comment_text_size = df["comment_text"].apply(lambda x: len(str(x))).mean()
print(avg_comment_text_size)
