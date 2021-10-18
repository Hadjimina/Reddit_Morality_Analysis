import unittest
from helpers.helper_functions import *
from feature_functions.speaker_features import *
import helpers.globals_loader as globals_loader

class speaker_Features_Tests(unittest.TestCase):
    globals_loader.load_spacy()  
    
    def test_get_author_age_and_gender(self):
        texts_synthetic = ["I'm 20 F","my boyfriend (21, F)","my (22, F)","I'm 23, f","I'm f, 24","I'm 25","I'm 1, f"]
        texts_real = ["I (27f) have been dating my boyfriend (30m) for ", "I(45M) and my wife(46F) got married in December of 2019. I have a daughter Cheryl(16F)", "I am 35m" ]

        texts = texts_synthetic+texts_real
        expected = [ 
            [("author_age", 20),("author_gender", 1)], #synthetic
            [("author_age", -1),("author_gender", -1)], 
            [("author_age", 22),("author_gender", 1)], 
            [("author_age", 23),("author_gender", 1)], 
            [("author_age", 24),("author_gender", 1)], 
            [("author_age", 25),("author_gender", -1)], 
            [("author_age", 1),("author_gender", 1)], 

            [("author_age", 27),("author_gender", 1)], #real
            [("author_age", 45),("author_gender", 0)], 
            [("author_age", 35),("author_gender", 0)], 
        ]
        
        
        for i in range(len(texts)):
            print(texts[i])
            act = get_author_age_and_gender(texts[i])
            exp = expected[i]
            self.assertEqual(act, exp)

    




    