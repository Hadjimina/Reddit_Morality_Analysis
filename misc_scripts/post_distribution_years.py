import pandas as pd
from datetime import datetime
import collections
import constants

df = pd.read_csv(constants.POSTS_CLEAN)
utc = df['post_created_utc'].tolist()
utc = [x for x in utc if str(x) != 'nan']
utc_date = list(map(lambda x: datetime.fromtimestamp(x).year, utc))
occurrences = collections.Counter(utc_date)
print(occurrences)