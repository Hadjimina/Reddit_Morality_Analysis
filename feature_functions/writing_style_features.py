from helpers.helper_functions import dict_to_feature_tuples
import logging as lg
import re
import coloredlogs
import constants as CS
import helpers.globals_loader as globals_loader

from helpers import *
coloredlogs.install()


def get_punctuation_count(post_text,**kwargs):
    """Count how many times certain punctuations syombols occur in text

    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
        [(str, int)]:  e.g. ("!_count": 10)
    """    

    symbols = ["!",'"', "?"]
    symbol_dict = dict.fromkeys(symbols, 0)

    #filter out hyperlinks
    #print(post_text)
    post_text = re.sub(r'http\S+', '', post_text)

    
    for i in range(len(symbol_dict.keys())):
        
        symbol = list(symbol_dict.keys())[i]
        symbol_dict[symbol] = post_text.count(symbol)

    tuple_list = dict_to_feature_tuples(symbol_dict, "_count")
    #tuple_list = [("{0}_count".format(k), v) for k, v in symbol_dict.items()]
    return tuple_list

def get_feats_dict(st_word):
    """Generate dictionary from stanze word.feats string

    Args:
        st_word (str): Stanza word.feats string

    Returns:
        dict: dict with all features as key and corresponding values
    """

    pairs_split = st_word.split("|")
    k = []
    v = []
    for pair in pairs_split:
        kv_split = pair.split("=")
        k.append(kv_split[0])
        v.append(kv_split[1])
    
    feats_dict = dict(zip(k, v)) if len(k) > 0 else {}
    return feats_dict

def get_tense_time_and_voice(post_text):
    """Iterate through text and count how many sentances are in past, present, future tense and in active/passive voice
       Return values are in absolute number and in percentage of #verbs / #all sentances (Where verbs are defined as "VERBS" from POS tagger)


    Args:
        post_text (str): Full body text of r/AITA post

    Returns:
         [(str, int)]:  e.g. [("future_count": 10), ("future_perc": 0.10)]
    """
    
    #post_text = "I do. I am doing. I have been doing. I did. I was doing. I had done. I had been doing. I will do. I will be doing. I will have done. I will have been doing. He will have had time."
    #post_text = "Monkeys adore bananas. Bananas are adored by monkeys."
    #post_text = "So, my (27F) cat died.To say I'm heartbroken is an understatement. It happened yesterday, in a terrible accident no one could have prevented that took the life of the most amazing best friend I have ever had. He has only two years and a half but I loved him with all I had, and I've been having the worst terrible day trying to accept the reality of him being gone. I apologize if i say something that doesn't make sense, English is not my first language. So, he died. It happened on a Sunday, around 11 am. I buried him at 2 pm on my backyard. At 5 pm, my sister (37F) and niece (7F) left home. At 22 pm, they came back with two kittens. Said they were orphaned and needed someone to take care of them. I, at first, thought they were in for foster care, but once they started to talk about naming them I started to get suspicious. This morning (Monday) I got out of bed to see if my gut feeling was right. I asked her when are they leaving?, like implying when would the foster care end, and she said no, they're here to stay.I immediately said I do not want him. They brought two, one for my niece and the other one for me, supposedly, without even asking. She asked why and I told her it's not even been 24 hours since I buried my cat, I'm not ready and on top of that, I dont like short haired, stripped cats (something irrelevant, trust me). I told her I knew her intentions were good but I'm simply not feeling that cat. She went off. She started saying she was disappointed, couldn't believe I'm rejecting a cat just because of the way it looks, and that she doesnt understand why I said I'll adopt one on my own once I'm ready. She kept insisting im rejecting him solely because of the way he looks. I love cats, but I can not feel a single thing for this one. I truly am not ready at all.I've always been picky with my pets. I dont look for anything but a connection between us. It hurts me to hear her say I'm rejecting him because of the way he looks, when she knows my cat (the one from the beginning) was a black cat I chose over his orange sibling just because I felt a better connection between us and I fucking love orange cats. If it was something appearance-based i would have gone with the orange one but I felt something in him, and I truly hit jackpot with him because I couldn't have chosen a better friend for these last two years. I hope you're resting easy, Kuro, I miss you so incredibly much already. If you have pets please give them a kiss right now, you really don't know when it's the last time you're going to be able to show them how much you love them. Its my first post here, but this is really eating me alive. My sister wont talk to me. AITA, Reddit?"
    
    #stz_nlp = kwargs["stz_nlp"]
    
    doc = globals_loader.nlp(post_text)    
    #for token in doc:
    #    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    
    tenses = ["past", "present", "future"]
    tense_dict = dict.fromkeys(tenses, 0)
    voices = ["active", "passive"]
    voice_dict = dict.fromkeys(voices, 0)
    verb_count = 0
    
    for sentence in doc.sents:
        #print("--NEW SENT--")
        #verb_count_old = verb_count
        sentence_voice = ""
        sentence_tense = ""

        voice_flag = False
        for token in sentence:
            # We only look at verbs

            if token.pos_ == CS.SP_VERB:
                feat_dict = get_feats_dict(str(token.morph))
                if "Voice" in feat_dict.keys():
                    print(feat_dict)

                # Do tense
                if CS.SP_FEATS_TENSE in feat_dict:
                    verb_count +=1

                    tense = feat_dict[CS.SP_FEATS_TENSE]
                    if tense == CS.SP_TENSE_PAST:
                        sentence_tense = "past"
                    elif tense == CS.SP_TENSE_PRESENT:
                        sentence_tense = "present"
                    elif tense == CS.SP_TENSE_FUTURE:
                        sentence_tense = "future"
                    tense_dict[sentence_tense] += 1


             # Do voice
        # TODO: This is definetly not perfect. Naive implementation only https://stackoverflow.com/questions/19495967/getting-additional-information-active-passive-tenses-from-a-tagger
            if voice_flag:
                continue

            if "nsubjpass" == token.dep_:
                sentence_voice = "passive"
            elif "nsubj" == token.dep_:
                sentence_voice = "active"

            if not sentence_voice == "":
                voice_flag = True
                voice_dict[sentence_voice] += 1

       
       
                    

                            
    # Merge & get percentage values
    merged_dict = {**tense_dict, **voice_dict}

    mrg_keys = [s + "_perc" for s in merged_dict.keys()]
    mrg_values = list(map(lambda x: round(x/verb_count,3), merged_dict.values()))
    perc_dict = dict(zip(mrg_keys, mrg_values))

    abs_features = dict_to_feature_tuples(merged_dict, suffix= "_abs")
    perc_features = dict_to_feature_tuples(perc_dict)
    
    return abs_features+perc_features
    
    