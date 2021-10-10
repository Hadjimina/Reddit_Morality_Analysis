import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import helpers.globals_loader as globals_loader
import gensim
from gensim.utils import simple_preprocess
import string

#alternatively can be done using spacy?
#TODO: add remove names option
def get_clean_txt(post_text, 
                    remove_URL=True,
                    remove_punctuation=False,
                    remove_newline=True,
                    merge_whitespaces=True,
                    do_lowercaseing=True,
                    remove_stopwords=False,
                    do_lemmatization=True):
    if not isinstance(post_text, str):
        type(post_text)

    if remove_URL:
        post_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(post_text))

    if remove_punctuation:
        post_text = post_text.translate(str.maketrans(' ', ' ', string.punctuation))

    # \n = newline, \r = carriage return
    if remove_newline:
        post_text = post_text.replace('\n', ' ').replace('\r', '')

    if merge_whitespaces:
        post_text = ' '.join(post_text.split())

    if do_lowercaseing:
        post_text = post_text.lower()

    if remove_stopwords: # removes things like [i, me, my, myself, we, our, ours, ...
        post_text = " ".join([word for word in post_text.split() if word not in stopwords.words('english')])
        

    if do_lemmatization:
        return globals_loader.nlp(post_text) #spacy
    else:
        return post_text

