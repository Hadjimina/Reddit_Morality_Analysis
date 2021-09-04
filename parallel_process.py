from posixpath import join
import threading
import logging as lg
import pandas as pd
import constants as CS

from multiprocessing import Process

class parallel_process (Process):
    def __init__(self, queue, thread_id, sub_df, features_to_generate):
        """Initialize Threading

        Args:
            threadID (int): id i.e. number of thread
            start_id (int): start index of subsection of post dataframe for this thread (inclusive)
            end_id (int): end index of subsection of post dataframe for this thread (exclusive)
            features_to_generate ({ string: [ ((int)->Dataframe, int) ]}): Dictionary wich contains function, 
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

        lg_str = "Using {0} threads".format(CS.NR_THREADS)
        if CS.NR_THREADS < 2:
            lg.warn(lg_str+" !")
        else:
            lg.info(lg_str)
            
        #print("DF on thread "+str(thread_id))
        

    def feature_to_df(thread, category, column, funct):
        """Generate dataframe out of return value of category name, column data and feature function

        returns : dataframe
            dataframe with headers corresponding to category and feature function 
            e.g "speaker" + "author_age" = "speaker_author_age",
            values are int returned from feature function

        Parameters
        ----------
        category : str
            Which feature category we are currently using

        column: [str]
            dataframe column we apply the feature function to

        funct: str->[(str, count)]
            feature function

        """
        lg.info('Running "{0}" on thread {1}'.format(funct.__name__, thread.thread_id))
    
        temp_s = column.apply(funct) # [(a,#)....]# was progress.apply
        #temp_s = column.parallel_apply(funct)
        fst_value = temp_s.iat[0]
        cols = ["{0}_{1}".format(category,tpl[0]) for tpl in fst_value]
        temp_s = temp_s.apply(lambda x: [v for s,v in x])
        df = pd.DataFrame(temp_s.to_list(), columns=cols)
        return df

    def run(self):
        """Apply all functions of "features_to_generate"
            to each row of subsection of dataframe
        """        
        #df_posts = globals_loader.df_posts
        #df_posts_subsection = df_posts.loc[self.start_index : self.end_index]

        # Create a list of all individual feature dfs and merge. Lastly append last column with post_id
        feature_df_list = []
        for category in self.features_to_generate:
            # We iterate over every column in dataset s.t. we first use all columns that use e.g. "post_author", before moving on
            for i in range(len(self.sub_df.columns)):
                for feature_tuple in self.features_to_generate[category]:
                    funct = feature_tuple[0]
                    idx = feature_tuple[1]
                    if idx == i:
                        col = self.sub_df.iloc[:,idx]
                        feature_df_list.append(self.feature_to_df(category, col, funct))
                        tmp_df = pd.concat(feature_df_list, axis=1)                
                        tmp_df.index = self.sub_df.index
                        tmp_df.to_csv(CS.TMP_SAVE_DIR+"/thread_{0}_tmp.csv".format(self.thread_id))

        
        # Post pends some post specific information
        post_pend = self.sub_df[CS.POST_PEND]
        post_pend.reset_index(drop=True, inplace=True)
        feature_df_list.append(post_pend)
        
        self.df = pd.concat(feature_df_list, axis=1, join="inner")
        self.df.index = self.sub_df.index
        
        self.queue.put(self.df)
        

            