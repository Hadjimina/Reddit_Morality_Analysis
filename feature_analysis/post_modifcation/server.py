from flask import Flask, render_template, request
app = Flask(__name__)
from collections import Counter
import liwc
import re
import pandas as pd

LIWC_PATH = "./data/liwc.dic"
MF_PATH = "./data/mf.dic"

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

def analyseLIWC(post_text, dict_path):
    print("ANALYSING")
    parse, category_names = liwc.load_token_parser(dict_path)
    result = Counter(category for token in tokenize(post_text) for category in parse(token))
    result_dict = dict(result)
    result_val = list(map(lambda x: [x], list(result_dict.values())))
    #modify keys
    
    if dict_path == LIWC_PATH:
        result_key = list(map(lambda x: "liwc_"+x.split("(")[0][:-1], list(result_dict.keys())))
    else:
        result_key = list(map(lambda x: "foundations_"+x, list(result_dict.keys())))

    result_dict = dict(zip(result_key, result_val))
    print(category_names)
    print(result_dict)
    df = pd.DataFrame.from_dict(result_dict)
    print(df)
    return df

def getFeatureValues(post_text):
    #analyseLIWC(post_text, LIWC_PATH)
    analyseLIWC(post_text, MF_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    data = []
    if request.method == 'POST':
        if request.form['submit_posts'] == 'Analyze':
            getFeatureValues(request.form['old_post'])
            data = [
                {
                    "ahr_old": 0.6,
                    "ahr_new": 0.6,
                    "changedFeatures": [
                        {"name": "feat1",
                         "value": 2,
                         "perc_change": 0.1},
                        {"name": "feat2",
                         "value": 3,
                         "perc_change": -0.1},
                    ]
                }
            ]
        else:
            pass  # unknown
    return render_template('index.html', data=data)


app.run(host='0.0.0.0', port=3001)
