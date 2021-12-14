from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from feature_functions.topic_modelling import *

feature_functions = {
    "minify": False,
    "title_handling": 1,  # 0 = prepend, 1 = standalone, 2 = first standalone, rerun as prepend
    "telegram_notify": True,
    "verbose": True,
    "hosts":
        {
            "phdesktop": {
                "skip": True,
                "username": "philipp",
                "host_address": "main",
                "path": "/home/philipp/Documennts/Coding",
                "mp": {
                    "speaker": [
                        #(get_author_amita_post_activity, CS.POST_AUTHOR),
                        #(get_author_info, CS.POST_AUTHOR), #slow
                        (get_author_age_and_gender, CS.POST_TEXT)
                    ],
                    "writing_sty": [
                        (get_punctuation_count, CS.POST_TEXT),
                        (get_emotions, CS.POST_TEXT),           
                        (aita_location, CS.POST_TEXT),          
                        (get_profanity_count, CS.POST_TEXT),    
                        #(check_wibta, CS.POST_TEXT)             

                    ],
                    "reactions": [
                        # (check_crossposts, CS.POST_ID),  # slow
                        #(get_judgement_labels, CS.POST_ID)
                    ]
                },
                "mono": {
                    "writing_sty": [
                        (get_spacy_features, CS.POST_TEXT),  # => 4h for 10%
                    ],
                    "reactions": [

                    ]
                },
                "spacy": [
                    get_tense_in_spacy,
                    get_voice_in_spacy,
                    get_sentiment_in_spacy,
                    get_focus_in_spacy,
                    get_emotions_self_vs_other_in_spacy,
                    get_profanity_self_vs_other_in_spacy,
                ],
                "topic": False,
                "foundations": True,
                "liwc": True,
                "reddit_instance_idx": 0
            },
             "104-171-249-216": {
                "skip":False,
                "username": "ubuntu",
                "host_address": "104.171.249.216",
                "path": "/home/ubuntu",
                "upload": True,
                "mp": {
                    "speaker": [
                    ],
                    "writing_sty": [
            
                    ],
                    "reactions": [
                    ]
                },
                "mono": {
                    "writing_sty": [
                    ],
                    "reactions": [
                    ]
                },
                "spacy": [
                ],
                "topic": True,
                "foundations": False,
                "liwc": False,
                "reddit_instance_idx": 0
             },
            "digitalocean-1": {
                "skip":True,
                "username": "root",
                "host_address": "64.227.39.41",
                "path": "/home/root/",
                "update_pip":False,
                "upload": True,
                "mp": {
                    "speaker": [
                        (get_author_info, CS.POST_AUTHOR),  # slow
                    ],
                    "writing_sty": [#
            
                    ],
                    "reactions": [
                        #(check_crossposts, CS.POST_ID),  # slow
                    ]
                },
                "mono": {
                    "writing_sty": [
                    ],
                    "reactions": [
                    ]
                },
                "spacy": [
                ],
                "topic": False,
                "foundations": False,
                "liwc": False,
                "reddit_instance_idx": 0
            },
            "phmedia": {
                "skip":True,
                "username": "philipp",
                "host_address": "phmedia.duckdns.org",
                "path": "/home/philipp/scripts/msc",
                "upload": True,
                "update_pip":False,
                "mp": {
                    "speaker": [
                    ],
                    "writing_sty": [
            
                    ],
                    "reactions": [
                        (check_crossposts, CS.POST_ID),  # slow
                    ]
                },
                "mono": {
                    "writing_sty": [
                    ],
                    "reactions": [
                    ]
                },
                "spacy": [
                ],
                "topic": False,
                "foundations": False,
                "liwc": False,
                "reddit_instance_idx": 1
            },

            # phmedia
        }
}
