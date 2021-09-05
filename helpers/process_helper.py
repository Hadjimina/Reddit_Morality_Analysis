
import pandas as pd
import constants as CS
import logging as lg
import stanza
from tqdm import tqdm

tqdm.pandas()

def process_run(feat_to_gen, sub_df, id):
    """ Apply all functions of "features_to_generate"
        to each row of subsection of dataframe
    """  

    #df_posts = globals_loader.df_posts
    #df_posts_subsection = df_posts.loc[self.start_index : self.end_index]

    # Create a list of all individual feature dfs and merge. Lastly append last column with post_id
    feature_df_list = []
    for category in feat_to_gen:
        # We iterate over every column in dataset s.t. we first use all columns that use e.g. "post_author", before moving on
        for i in range(len(sub_df.columns)):
            for feature_tuple in feat_to_gen[category]:
                
                funct = feature_tuple[0]
                idx = feature_tuple[1]
                if idx == i:
                    col = sub_df.iloc[:,idx]
                    #feature_df_list.append(self.feature_to_df(category, col, funct, self.stz_nlp))
                    feature_df_list.append(feature_to_df(id, category, col, funct)) # we only pass stanze refernce in mono processing
                    tmp_df = pd.concat(feature_df_list, axis=1)                
                    tmp_df.index = sub_df.index
                    tmp_df.to_csv(CS.TMP_SAVE_DIR+"/thread_{0}_tmp.csv".format(id))

    
    # Post pends some post specific information
    
    if id == CS.MONO_ID:
        post_pend = sub_df[CS.POST_PEND_MONO]
    else:
        post_pend = sub_df[CS.POST_PEND]
    #TODO: here we assume that order is kept and never switched up => also pass ids into process_run and merge based on id in the end to be sure
    post_pend.reset_index(drop=True, inplace=True)
    feature_df_list.append(post_pend)
    return feature_df_list


def feature_to_df(id, category, column, funct):#stz_nlp
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
        lg.info('Running "{0}" on thread {1}'.format(funct.__name__, id))


        #temp_s = column.apply(funct, stz_nlp=stz_nlp) # [(a,#)....]# was progress.apply
       
        temp_s = column.progress_apply(funct)
        
        fst_value = temp_s.iat[0]
        cols = ["{0}_{1}".format(category,tpl[0]) for tpl in fst_value]
        temp_s = temp_s.apply(lambda x: [v for s,v in x])
        df = pd.DataFrame(temp_s.to_list(), columns=cols)
        return df