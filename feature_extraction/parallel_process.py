from posixpath import join
import threading
import logging as lg
import pandas as pd
import constants as CS
from helpers.process_helper import *

from multiprocessing import Process

class parallel_process (Process):
    def __init__(self, queue, thread_id, sub_df, features_to_generate, ):
        """Initialize Threading

        Args:
            threadID (int): id i.e. number of thread
            start_id (int): start index of subsection of post dataframe for this thread (inclusive)
            end_id (int): end index of subsection of post dataframe for this thread (exclusive)
            features_to_generate ({ string: [ ((int)->Dataframe, int) ]}): Dictionary which contains function, 
                argument tuples

        """        
        #threading.Thread.__init__(self)
        super(parallel_process, self).__init__()
        self.queue = queue
        #self.idx = idx
        
        self.thread_id = thread_id
        self.sub_df = sub_df
        self.features_to_generate = features_to_generate
        self.df = None
    
        

    def run(self):
        """Apply all functions of "features_to_generate"
            to each row of subsection of dataframe
        """         
        feature_df_list = process_run(self.features_to_generate, self.sub_df, self.thread_id)
        self.df = pd.concat(feature_df_list, axis=1, join="inner")
        self.df.index = self.sub_df.index
        
        self.queue.put(self.df)
        

            