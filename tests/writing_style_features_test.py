import unittest
from helpers.helper_functions import *
from  feature_functions.writing_style_features import *
from  feature_functions.speaker_features import *
import helpers.globals_loader as globals_loader

class writing_Style_Tests(unittest.TestCase):
    globals_loader.load_spacy()  
        
    def test_get_punctuation_count(self):
        texts = ['!?"',
                 'adsf öa?!??asfpa ! asdi pu p"9823r  a?__"l öasd""', 
                 """My roommate has been helping me get in shape. For background, I've been pretty sedentary for a few years now, so I am very out of shape and very overweight. I appreciate her help, but she is like a drill instructor. I didn't like them when I was enlisted, and I don't like them now. Last night, I was having a bad day because I didn't get any sleep the night before. So, after working 9 hours, granted I work sitting down but still, at 1030p, we went downstairs to the gym. We had been doing the elliptical but none were available. So, we decided I'd walk (torn acl, no running). She started fiddling with the settings, next thing I know, it's been 25 mins, and I've been walking uphill at a brisk pace for the whole time. But that wasn't "making me sweat" enough so 25-30 mins in, she turned the incline up. My speed was already as fast as it could be without running. After a mile and a half she was like, okay you want to stop, it was about 35-40 mins in. I said, yeah but I need to cool down. She agreed, so I turned it to about half the speed I was doing and got rid of the incline. Here's where the problem happened. She was like don't stop, and I said I'm not, I'm still walking. I tried to explain that I was tired, but she was like, you're not even doing anything! Then she starts in on how I have a bad attitude, and I don't listen. And how I'm lazy and never do anything and can't even walk without complaining. So I was like, fine, I'll keep going. I turned it back up, and she freaked out saying I was being passive aggressive by turning it back up. I got angry, I was like first I was going too slow, now I'm going too fast. I'm not psychic I don't know what magic number is right to you, you're making this harder than it needs to be. Then she starts screaming at me in front of the other 2 people in there. So, I stopped the treadmill and was like fuck this, I'm out. She started going toward the door, waiting for me to follow her and I was like just go, I'm not coming with you. Then she said we'll I'm not leaving you can't tell me what to do. So, I walked out the other direction. Who is the asshole? I feel like I was tired, she knew that. I told her I didn't have the energy for a hard workout before we went down there. I also feel like she doesn't give a crap how I feel or what's going on with me. It's like unless I can't stand up after exercising, I haven't worked hard enough. I feel like yes, I'm just walking, but after doing nothing for so long, and considering my weight, walking is exercising. The previous 3 nights after cardio, I did some core and arm work, so it's not like I didn't do anything all week. We have exercised 4 nights in a row, I feel like that's pretty good. Especially considering my previous in a row streak was 2 days. It's like no matter what I do I can't live up to this standard that keeps changing. Idk what to do, so I'm crowd sourcing. Please help."""]

        expected = [ 
            [("!_count",1),('"_count',1),("?_count",1)], 
            [("!_count",2),('"_count',4),("?_count",4)], 
            [("!_count",1),('"_count',2),("?_count",1)] 
        ]
        
        self.checker(texts, expected, get_punctuation_count)
    
    def test_aita_location(self):
        texts = ["Everyone in the restaurant looked at me funny but I didn’t care and just enjoyed my chicken fries. When we got home my mom said I embarrassed them in the restaurant by getting fast food delivered. But she didn’t stop for me and I paid for the delivery myself. AITA?", 
                      "And if that had been all I would have attended, but later my brother (m40) called me and accused me of being envious and told me that I just wanted attention and that's why I got pregnant again, to overshadow his wife. And a few more things that broke my heart, I couldn't stop crying and my husband who was with me hung up the phone for me and told me not to go, that it was enough and that I should cut them off, so for the first time I did the what he said and that didn't work because now I'm the villain. They said that my sister-in-law spent years trying to conceive and now that she finally got pregnant I was ruining everything, that she cried and that she didn't know why I didn't go, we have a really good relationship but I didn't told her anything because I didn't want to create problems between her and my brother, so I wanted to know, am I the asshole for not going?",
                      "",
                      "Aita, Day of the wedding comes, all of us bridesmaids and Mary meet early in the morning for pictures outside the chapel. am i the ah? Mary sees my necklace and loves it, and actually asked if she could switch her necklace with mine. Some other bridesmaids chime in and say that it would be her “something borrowed”. I tried as politely as I could to tell her that my fiancé got this for me to wear to the wedding and especially since he can’t be here, I’d like to keep it on. Mary and the other bridesmaids were persistent, saying it was her wedding and her pictures and I wasn’t being accommodating, but I firmly told them no. Mary then said if I could at least take it off since it looked nicer than her own jewelry. Again, I told her I’d rather not, I’d like to share the wedding photos of myself in the necklace with my fiancé. She was not happy, neither of the bridesmaids were either. I received a lot of cold shoulders and dirty looks at the ceremony and reception. I felt awful. After the first dance, Mary’s sister came up to me to tell just how upset Mary was and rude it was that I upset her so on her big day.Am I an asshole? It’s been more than a week and I’m still thinking about it nonstop and Mary hasn’t answered any of my texts."
        ]
        
        expected = [ 
            [("aita_count",1), ("aita_avg_location",260/265), ("aita_fst_location",260/265), ("aita_lst_location",260/265) ],
            [("aita_count",1), ("aita_avg_location",850/881), ("aita_fst_location",850/881), ("aita_lst_location",850/881)],
            [("aita_count",0), ("aita_avg_location",-1), ("aita_fst_location",-1), ("aita_lst_location",-1) ],
            [("aita_count",3), ("aita_avg_location",((1122/1247+122/1247)/3)), ("aita_fst_location",0.0), ("aita_lst_location",1122/1247) ],
           
        ]

        self.checker(texts, expected, aita_location)

    def test_check_wibta(self):
        texts = ["to have lives and hobbies outside of work, but I just think this is way too inappropriate to go unpunished. She can’t just run away from her poor and unprofessional behavior. If she’s going to be part of this profession, she should be more mindful of the content she puts on the internet. And I honestly don’t think other parents would be pleased to know that their kids have a teacher that shouts loudly on the internet about which anime character she wants to “toss her salad. WIBTA if I show her new school administration her twitch account?", 
                 "What the actual fuck. So i convinced myself to terminate the therapy process. Would i be the asshole if i do this?", 
                 "I was livid at my wife. This was the 5th interview that she ruined like this. I told her that I give up on trying to make her life easier and that I'm not ready to go looking for any more jobs since she doesn't even want to maintain silence during important interviews. I told her to start working again after the birth and that I am ok paying for a nanny/babysitter.She said that in being unreasonable in expecting perfect silence at home.AITA?", 
                 "When most of the relatives left, Amy’s parents came in my room and told me I was blowing things out of proportion (and apparently the relatives thought that too). Amy was on my side, but her parents were quite angry at me for several days, and almost got me kicked out of their house.So, tell me, WITAH?"]

        expected = [ 
            [("is_wibta",1)], 
            [("is_wibta",1)], 
            [("is_wibta",0)], 
            [("is_wibta",1)], 
        ]

        self.checker(texts, expected, check_wibta)
            
    def test_get_profanity_count(self):
        texts = ["asshole, ass, ah, hole, bitch", 
                       "I think YTA", 
                       "am I the asshole"]
        
        expected = [
            [("profanity_abs", 3),("profanity_norm",3/5)],
            [("profanity_abs", 0),("profanity_norm",0)],
            [("profanity_abs", 1),("profanity_norm",1/4)],
        ]
        
        self.checker(texts, expected, get_profanity_count)
        
    
    def checker(self, texts, expected, fn):
        for i in range(len(texts)):
            print(texts[i])
            act = fn(texts[i])
            exp = expected[i]
            self.assertEqual(act, exp)
            
   
