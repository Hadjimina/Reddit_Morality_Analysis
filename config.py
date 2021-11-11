from feature_functions.reaction_features import *
from feature_functions.speaker_features import *
from feature_functions.writing_style_features import *
from feature_functions.topic_modelling import *

feature_functions = {"phdesktop": {
    "mp": {
        "speaker": [
            #(get_author_amita_post_activity, CS.POST_AUTHOR),
            (get_author_info, CS.POST_AUTHOR),
            (get_author_age_and_gender, CS.POST_TEXT)
        ],
        "writing_sty": [
            (get_punctuation_count, CS.POST_TEXT),
            (get_emotions, CS.POST_TEXT),
            (aita_location, CS.POST_TEXT),
            (get_profanity_count, CS.POST_TEXT),
            (check_wibta, CS.POST_TEXT)

        ],
        "reactions": [
            # (check_crossposts, CS.POST_ID),  # slow
            (get_judgement_labels, CS.POST_ID)
        ]
    },
    "mono": {
        "writing_sty": [
            (get_spacy_features, CS.POST_TEXT),  # => 4h for 10%
        ],
    },
    "spacy": [
        get_tense_in_spacy,
        get_voice_in_spacy,
        get_sentiment_in_spacy,
        get_focus_in_spacy,
        get_emotions_self_vs_other_in_spacy,
        get_profanity_self_vs_other_in_spacy,
    ],
    "topic": True,
    "foundations": False,
    "liwc": False
}
}