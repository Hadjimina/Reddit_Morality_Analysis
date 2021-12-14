import pandas as pd
from datetime import datetime
import collections
import sys
sys.path.insert(0,'../..')
import constants


df = pd.read_csv(constants.POSTS_CLEAN)
utc = df['post_created_utc'].tolist()
utc = [x for x in utc if str(x) != 'nan']
utc_date = list(map(lambda x: datetime.fromtimestamp(x).year, utc))
occurrences = collections.Counter(utc_date)

# Get values in percentages
occurrences_values = list(occurrences.values())
occurrences_keys = list(occurrences.keys())

total_posts = sum(occurrences_values)
fn = lambda x: round(x/total_posts,2)

perc_values = list(map(fn,occurrences_values))
zip_iterator = zip(occurrences_keys, perc_values)
perc_dict = dict(zip_iterator)

print("Absolute values")
print(occurrences)
print("Percentages")
print(perc_dict)
