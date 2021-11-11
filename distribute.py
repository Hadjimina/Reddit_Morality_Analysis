import socket
import config as config
import constants as CS
import create_features

def set_feature_functions():
    hostname = socket.gethostname().lower()    
    
    #CS.USE_MINIFIED_DATA = True
    #CS.FEATURES_TO_GENERATE_MP = config.feature_functions[hostname]["mp"]
    #CS.FEATURES_TO_GENERATE_MONO = config.feature_functions[hostname]["mono"]
    #CS.SPACY_FUNCTIONS = config.feature_functions[hostname]["spacy"]
    #CS.DO_TOPIC_MODELLING = config.feature_functions[hostname]["topic"]
    #CS.LOAD_FOUNDATIONS = config.feature_functions[hostname]["foundations"]
    #CS.LIWC = config.feature_functions[hostname]["liwc"]
    
if __name__ == "__main__":
    set_feature_functions()
    create_features.main([])
    