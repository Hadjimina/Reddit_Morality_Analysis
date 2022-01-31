import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import sys
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
#from mrmr import mrmr_classif
import matplotlib.pyplot as plt
from pprint import pprint
#import tensorflow as tf
#from autokeras import StructuredDataClassifier
import xgboost as xgb
import shap
import gc
import json
from itertools import islice
import functools
from scipy import stats
from datetime import date
from tqdm import tqdm
#%matplotlib inline
#import mpld3
#mpld3.enable_notebook()
# rf, pps (https://github.com/8080labs/ppscore), correleation, shapele, mrmr

params = {
    "norm_vals": [0,1,2],               # normalised: 0 = only "abs", 1 = only "norm", 2 = norm and abs
    "weighted_vals" : [True, False],     # weighted_vals: whether votes should be weighted by comment score FALSE IS BETTER 0.32
    "title_prep_vals" : [True, False],   # title_prepend: whether to use the title prepended or standalone dataset
    "sampling_vals" : ["up", "down"],   # sampling_vals: which type of sampling should be done
    "topics_separate": False,           # if each topic should be analysed separately
    "predict":"ratio",                  # should we predict "class" (classification for binary) or "ratio" (regression for AHR)
    "mapping_type":[  "opposite", "clip"], # should we "clip" negative votes or map them to the "opposite"
    "ratio": [0.5,0.3,0.1  ],      # which most extreme AHR or YTA_ratio we want to predict 0.3, 0.2, 0.1, 0.05
    "wo_metadata": False #wheter we should include metadata columns (e.g. post_score, account_karam, link_karma) set MANUALLY
}

FORCE_SIMPLIFY = False
DO_SHAPLY = True
SHOW_PREDICTION_DISTRIBUTION = False
FEAT_IMPORTANCE_N = 50
OUTPUT_DIR = "./output/" if not params["wo_metadata"] else "./output_wometadata/"
DATASETS_DIR = "./datasets/"
    
def get_data(normalised=1, weighted=True, title_prepend=True, topics_separate=False, wo_metadata=False):
    prepend_csv = "prepend_mf_liwc_angel_info_topic_scores_reactions_reduced_da.csv"
    standalone_csv = "standalone_liwc_mf_angel_info_topic_scores_reduced_reactions_da.csv"
    
    if title_prepend:
        df = load_wo_cols(DATASETS_DIR+prepend_csv, wo_metadata=wo_metadata)
    else:
        df = load_wo_cols(DATASETS_DIR+standalone_csv, wo_metadata=wo_metadata)
         

    if normalised < 2:
        df = df[df.columns.drop(list(df.filter(regex="_abs" if normalised == 1 else "_norm")))]
        

    keys = ["info", "yta", "nah", "esh", "nta"]
    weight = "weighted_" if weighted else ""
    values = ["reactions_"+weight+k.upper() for k in keys]
    acros = dict(zip(keys, values))
    
    dfs = []
    if topics_separate >0:
        
        topic_min = df["topic_nr"].min()
        topic_max = df["topic_nr"].max()
        #print(f"Data split by topic ({topic_min}, {topic_max})")
         
        for i in range(topic_min, topic_max+1):
            dfs.append(df.loc[df["topic_nr"]==i])
    else:
        dfs = [df]

    #print(f"Number of dataframes: {len(dfs)}")

    
    return dfs, acros

def load_wo_cols(path, remove_cols=[],verbose=False, wo_metadata=False):
    cols_to_remove = ["post_text", "Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "Unnamed: 0.1", 
                      "Unnamed: 0.1.1", "liwc_post_id", "foundations_post_id", 
                      "foundations_title_post_id", "liwc_title_post_id", "post_created_utc"]+remove_cols
    metadata = ["speaker_account_comment_karma", "post_num_comments", "post_ratio", "speaker_account_age", "speaker_account_link_karma", "post_ups", "post_downs", "post_score","reactions_is_devil", "reactions_is_angel"]
    
    if wo_metadata:
        cols_to_read = cols_to_remove+metadata
        
    removed = []
    df = pd.read_csv(path, nrows=10)
    cols_to_read = list(df.columns)
    
    if verbose:
        print(cols_to_read)
    for col in cols_to_remove:
        if col in cols_to_read:
            cols_to_read.remove(col)
            removed.append(col)
    
    print(f"Removed {removed} from {path.split('/')[-1]}")        
    df = pd.read_csv(path, usecols=cols_to_read)
    return df



def sampling(y, kind="up", indices=[], verbose=False):
    
    df_y = pd.DataFrame(data={"Y":y})
    
    if len(indices)>0:
        if verbose:
            print(f"Using {len(indices)} indices")
    else:
        indices = range(len(indices))
        

    # Get list of indices for classes that are in the indices array
    c0_idx = pd.Series(df_y.loc[df_y["Y"]==0].index.values)
    c0_idx = c0_idx[c0_idx.isin(indices)]
    c1_idx = pd.Series(df_y.loc[df_y["Y"]==1].index.values)
    c1_idx = c1_idx[c1_idx.isin(indices)]
    
    if verbose:
        print(f"    Y=0: {c0_idx.shape}")
        print(f"    Y=1: {c1_idx.shape}")

    if kind == "up":
        #upsample
        if len(c0_idx)>len(c1_idx):
            n = len(c0_idx)
            c1_idx_sampeled = c1_idx.sample(n=n, random_state = 1, replace=len(c1_idx)<n).values
            c0_idx_sampeled = c0_idx.values
            if verbose:
                print(f"Upsampling Y=1 with {n} samples")
                
        elif len(c0_idx)<len(c1_idx):
            n = len(c1_idx)
            c0_idx_sampeled = c0_idx.sample(n=n, random_state = 1, replace=len(c0_idx)<n).values
            c1_idx_sampeled = c1_idx.values
            if verbose:
                print(f"Upsampling Y=0 with {n} samples")
                
    elif kind =="down":
        #downsample
        if len(c0_idx)>len(c1_idx):
            n = len(c1_idx)
            c0_idx_sampeled = c0_idx.sample(n=n, random_state = 1, replace=len(c0_idx)<n).values
            c1_idx_sampeled = c1_idx.values
            if verbose:
                print(f"Downsampling Y=0 with {n} samples")
        elif len(c0_idx)<len(c1_idx):
            n = len(c0_idx)
            c1_idx_sampeled = c1_idx.sample(n=n, random_state = 1, replace=len(c1_idx)<n).values
            c0_idx_sampeled = c0_idx.values
            if verbose:
                print(f"Downsampling Y=1 with {n} samples")

    all_idx = np.concatenate((c0_idx_sampeled, c1_idx_sampeled), axis=0)
    
    if verbose:
        df_tmp = df_y.iloc[all_idx]
        print(f"   Y=0: {len(df_tmp.loc[df_tmp['Y']==0])}")
        print(f"   Y=1: {len(df_tmp.loc[df_tmp['Y']==1])}")
    return all_idx

def obj_size_fmt(num):
    if num<10**3:
        return "{:.2f}{}".format(num,"B")
    elif ((num>=10**3)&(num<10**6)):
        return "{:.2f}{}".format(num/(1.024*10**3),"KB")
    elif ((num>=10**6)&(num<10**9)):
        return "{:.2f}{}".format(num/(1.024*10**6),"MB")
    else:
        return "{:.2f}{}".format(num/(1.024*10**9),"GB")


def memory_usage():
    memory_usage_by_variable=pd.DataFrame({k:sys.getsizeof(v)\
    for (k,v) in globals().items()},index=['Size'])
    memory_usage_by_variable=memory_usage_by_variable.T
    memory_usage_by_variable=memory_usage_by_variable.sort_values(by='Size',ascending=False).head(10)
    memory_usage_by_variable['Size']=memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
    return memory_usage_by_variable

def opposite_jdgmt(judg):
    
    if "NTA" in judg  :
        rtn =  judg.replace("NTA","YTA")
    elif "NAH" in judg:
        rtn =  judg.replace("NAH","ESH")
    elif "YTA" in judg:
        rtn =  judg.replace("YTA","NTA")
    elif "ESH" in judg:
        rtn =  judg.replace("ESH","NAH")
    elif "INFO" in judg:
        rtn = judg
    
    return rtn+"_neg_vals"

def get_vote_counts(df, acros):
    dct = {}
    for acr in list(acros.values()):
        dct[acr] = len(df[acr].to_numpy().nonzero()[0])
    
    dct["total"] = np.sum(list(dct.values()))
    print(dct)
    
    
# mapping is either "clip", meaning negative votes are just set to 0, or "oppossite", meaning we use the mapping table in "opposite_jdgmt"
def map_negative_values(df, acros, mapping="clip"):
    
    if mapping == "opposite" or mapping =="map":
        #print("Map = opposite")    
        for k in acros.keys():
            acr = acros[k]
            #print(f"{acr} pos amount {len(df.loc[df[acr] > 0])}")
            #print(f"{acr} neg amount {len(df.loc[df[acr] < 0])}")
            
            
            if k == "info":
                continue
            
            # create temporary columns containing zeros and only negative votes for each vote type (except info)        
            df[acr+"_neg_vals"] = 0
            df.loc[df[acr] < 0, acr+"_neg_vals"] = df[acr]*-1
            df.loc[df[acr] < 0, acr] = 0
            
        for k in acros.keys():
            if k == "info":
                continue
            acr = acros[k]
            #set negative values to 0 & add opposite judgement vote 
            df[acr] = df[acr] + df[opposite_jdgmt(acr)]
           
    elif mapping =="clip":
        #print("Map = clip")    
        for k in acros.keys():
            acr = acros[k]
            df.loc[df[acr]<0,acr]=0
        
            
    # finally set all negative info votes to 0
    #df[df[acros["info"]] < 0] = 0
    
    #print("info sum", df[acros["info"]].min())
    
    return df


def get_data_classes(df, acros, ratio=0.5, verbose=False, predict="class", judgement_weighted=True, mapping="clip"):
    if verbose:
        print(f"df original shape {df.shape}")
        
    n_rows_old = len(df)
    
    # Map negative judgements to opposing judgement
    # i.e. YTA<->NTA, ESH<->NAH
    if judgement_weighted:
        df = map_negative_values(df, acros,mapping=mapping)
    
    if predict=="class":   
        # We only look at YTA and NTA
        df["YTA_ratio"] = df[acros["yta"]]/(df[acros["info"]]+ df[acros["yta"]]+ df[acros["nah"]]+df[acros["esh"]]+df[acros["nta"]])

        # drop all rows where the majority is not YTA or NTA
        df = df.loc[((df[acros["yta"]] > df[acros["info"]]) & (df[acros["yta"]] > df[acros["nah"]]) & (df[acros["yta"]] > df[acros["esh"]])) | ((df[acros["nta"]] > df[acros["info"]]) & (df[acros["nta"]] > df[acros["nah"]]) & (df["reactions_weighted_NTA"] > df[acros["esh"]]))]
        if verbose:
            print(f"Drop all rows where majority is not YTA or NTA {df.shape}")

        #drop all rows that are not "extreme" enough
        df = df.loc[(1-ratio<=df["YTA_ratio"]) | (df["YTA_ratio"]<=ratio)]
        if verbose:
            print(f"Removed {n_rows_old-len(df)} rows b.c. not enough agreement. Now {df.shape}")

        #specifc classes & drop unnecesarry
        df["Y"] = np.where(df[acros["yta"]] > df[acros["nta"]], 1,  0) # YTA = Class 1, NTA = class 0
        if verbose:
            print(df.shape)
            
    elif predict == "ratio":
        # Y = asshole ratio(AHR) = (YTA+ESH)/(YTA+ESH+NTA+NAH)
        
        #drop posts w.o. votes
        tmp = df[acros["yta"]]+ df[acros["nah"]]+df[acros["esh"]]+df[acros["nta"]]
        tmp = tmp[tmp != 0]
        tmp = (df[acros["yta"]]+df[acros["esh"]])/tmp
        df["Y"] = tmp
        
        if verbose:
           print(f"Removed {n_rows_old-len(df)} rows b.c. no votes. Now {df.shape}")
        n_rows_old = len(df)
        df = df.loc[(1-ratio<=df["Y"]) | (df["Y"]<=ratio)]
        if verbose:
            print(f"Removed {n_rows_old-len(df)} rows b.c. not enough agreement. Now {df.shape}")
        
    if np.min(df["Y"]) < 0 or np.max(df["Y"])>1:
        raise Exception("Y value should be in range [0,1]")
        
    # get list of all columns that contain uppercase vote acronym
    vote_acroynms = list(filter(lambda x: any([acr.upper() in x for acr in list(acros.keys())]), list(df.columns)))  
    df = df.drop(columns=vote_acroynms)
    
    cols_to_drop =  ["post_text", "post_id", "post_created_utc"] 
    for col in cols_to_drop:
        if col in list(df.columns):
            df = df.drop(columns=[col])
        
    #print(df.info(memory_usage="deep"))

    # Removing top 4 most important features leads to 0.66 f1
    if verbose:
        print(df.shape)
    
    X = df.drop(columns=["Y"])
    y = df["Y"].to_numpy()

    feat_name_lst = list(X.columns)

    # scaling
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, feat_name_lst     

def main(args):
    args = " ".join(args)
    
    params["wo_metadata"] = ("wo" in args or "no" in args) and ("metadata" in args or "meta" in args)
    OUTPUT_DIR = "./output/" if not params["wo_metadata"] else "./output_wometadata/"
    
    print("CLASSIFICATION\n----" if params["predict"]=="class" else "REGRESSION\n----")
    if FORCE_SIMPLIFY:
        print("SIMPLIFYING")
        for k, v in params.items():
            if isinstance(v, list):
                params[k] = [v[0]]

    if params["predict"] == "ratio":
        params["sampling_vals"] = ["up"]

    #params["title_prep_vals"]= ["True", "False"]
 

    # mpc = MLPClassifier( random_state=1) seems pretty shitty
    # boost = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0,max_depth=20, random_state=0)
    xgboost = xgb.XGBClassifier(verbosity = 0) if params["predict"] == "class" else xgb.XGBRegressor(verbosity = 0)
    rfc = RandomForestClassifier(n_estimators= 766, min_samples_split= 2, min_samples_leaf= 1, max_features="auto", max_depth= 40, bootstrap= False, n_jobs=-1) if params["predict"] == "class" else RandomForestRegressor(n_estimators= 766, min_samples_split= 2, min_samples_leaf= 1, max_features="auto", max_depth= 40, bootstrap= False, n_jobs=-1)

    #classifiers = [(boost,"boost"), (xgboost,"xgboost"), (rfc,"rfc")]
    classifiers = [(xgboost, "xgboost"),]


    print("Models:\n"+"\n".join(["  "+str(s) for m,s in classifiers]))
    print("\nClassification params:\n"+str(json.dumps(params, indent = 4)))

    #cv = KFold(n_splits=SPLITS, random_state=1, shuffle=True)
    #for train, test in cv.split(X, y):
    test_scores = {}
    nr_samples = []
    class_ratio = [] #only used for classification
    top_n_features = {}
    
    #for correlation
    #y_lst = []

    current_iter = 1
    max_iter = functools.reduce(lambda a, b: a*b, [len(x) if isinstance(x, list) else 1 for x in list(params.values())])
    for norm in params["norm_vals"]:
        for weighted in params["weighted_vals"]:
            for title_prep in params["title_prep_vals"]:
            
                if "dfs" in locals() or "dfs" in globals():
                    for i in dfs:
                        del i
                if "df" in list(memory_usage().index):
                    del df
                    gc.collect()
                
                dfs, acros = get_data(normalised=norm, weighted=weighted, title_prepend=title_prep,
                                      topics_separate=params["topics_separate"], wo_metadata=params["wo_metadata"])

                for smp in params["sampling_vals"]:
                    for rto in params["ratio"]:
                        for mpt in params["mapping_type"]:
                            for df in dfs: 
                                df_cpy = df.copy()
                                X, y, feat_name_lst = get_data_classes(df_cpy, ratio=rto, acros=acros, 
                                                                       predict=params["predict"], judgement_weighted=weighted,
                                                                       mapping=mpt,verbose=False)    
                                #y_lst.append(y.flatten())
                                
                                # sanity check for y sum
                                y_sum_should = 35461 if mpt == "clip" else 35850
                                eps = np.absolute(np.sum(y) - y_sum_should)
                                if eps > 3 and rto == 0.5 and params["predict"]=="ratio":
                                    print(np.sum(y))
                                    #raise Exception("WRONG Y SUM FOR "+mpt+" ("+str(eps)+")")
                                    print("WRONG Y SUM FOR "+mpt+" ("+str(eps)+")")
                                
                                train, test = train_test_split(range(len(X)), test_size=0.33, random_state=42)

                                if params["predict"]=="class":
                                    #print("Doing sampling")
                                    train = sampling(y, kind=smp, indices=train, verbose=False)

                                for clf_tpl in classifiers:
                                    clf = clf_tpl[0]
                                    clf_name = clf_tpl[1]
                                    clf_name += f"_norm={norm}"
                                    clf_name += "_title=prep" if title_prep else "_title=standalone"
                                    clf_name += "_weighted" if weighted else ""
                                    clf_name += "_ratio="+str(rto)
                                    clf_name += "_"+mpt

                                    clf_name += "_topic_"+str(df["topic_nr"].iloc[0]) if params["topics_separate"] else ""                      
                                    clf_name += "_wometadata" if params["wo_metadata"] else ""

                                    print(f"Running ({current_iter}/{max_iter*len(dfs)}):\n  {clf_name}")
                                    #print(f"    Training set size: {len(train)} Testing set size: {len(test)} => TOTAL SIZE: {len(train)+len(test)}")
                                    X_train = X[train, :]
                                    y_train = y[train]
                                    X_test = X[test, :]
                                    y_test = y[test]

                                    if SHOW_PREDICTION_DISTRIBUTION:
                                        plt.hist(y[train], bins=10*32)
                                        plt.show()

                                    clf.fit(X_train, y_train)
                                    y_pred = clf.predict(X_test)
                                    
                                    if DO_SHAPLY:
                                        explainer = shap.explainers.GPUTree(clf, X_train)
                                        #explainer = shap.explainers.Tree(clf, X_train)
                                        shap_values = explainer(X_train)
                                        shap.summary_plot(shap_values, X_train, feature_names=feat_name_lst, max_display=50, show=False)
                                        f = plt.gcf()
                                        #f.set_size_inches(18.5, 10.5)
                                        plt.savefig(f'{OUTPUT_DIR}{clf_name}.png',bbox_inches = "tight")
                                        plt.clf()
                                        
                                        # save top N features
                                        feature_names = feat_name_lst
                                        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
                                        vals = np.abs(shap_df.values).mean(0)
                                        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
                                        shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                                        
                                        name_importance_zip = list(zip(
                                            shap_importance["col_name"][:FEAT_IMPORTANCE_N],
                                            shap_importance["feature_importance_vals"][:FEAT_IMPORTANCE_N]))+["RMSE ="+ str(metrics.mean_squared_error(y_test, y_pred, squared=False))]
                                        
                                        top_n_features[clf_name] = name_importance_zip
                                
                                        
                                
                                    # We have more Y=0 (NTA) than Y=1 (YTA)
                                    # metrics.plot_confusion_matrix(classify, X_test, y_test)  
                                    #plt.show()
                                    #print(metrics.classification_report(y_test, y_pred, target_names=["NTA (0)", "YTA (1)"]))

                                    if params["predict"] == "class":
                                        print("class")
                                        # testing score
                                        f1_test = metrics.f1_score(y_test, y_pred, average="weighted")
                                        acc_test = metrics.accuracy_score(y_test, y_pred)
                                        test_scores[clf_name]=f1_test
                                        nr_samples.append(len(X_train))
                                        ratio = y[train].sum()/len(y[train])
                                        if ratio < 0.5:
                                            ratio = (len(y[train])-y[train].sum())/len(y[train])
                                        class_ratio.append(ratio)
                                        current_iter+=1
                                        print(f"    Accuracy: {acc_test}\n    F1: {f1_test}")

                                    elif params["predict"] == "ratio":
                                        print("ratio")
                                        mean_abs = metrics.mean_absolute_error(y_test, y_pred)
                                        mean_sqr = metrics.mean_squared_error(y_test, y_pred)
                                        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
                                        
                                        
                                        
                                        test_scores[clf_name]=rmse
                                        nr_samples.append(len(X_train))
                                        current_iter+=1
                                        print(f"    Mean absolute: {mean_abs}\n    Mean squared: {mean_sqr}\n    Root Mean Squared: {rmse}")
                                    
                                    #plt.figure(2)
                                    #plt.scatter(y_pred,y_test,s=0.35)
                                    #plt.plot([0,1], color="red")
                                    #plt.plot([0.15,1.15], color="orange")
                                    #plt.plot([-0.15,0.85], color="orange")
                                    #plt.title("Prediction accuracy for "+params["predict"])
                                    #plt.xlabel("Predicted value")
                                    #plt.ylabel("Actual value")
                                    #plt.show()
                                    
                                    del df_cpy
                                
    
                        
    #test_scores = np.array(test_scores)
    #print(f"Average Test F1: {np.mean(test_scores[:,0])}, Test Accuracy: {np.mean(test_scores[:,1])}")

    classifiers = list(test_scores.keys())
    scores = list(test_scores.values())
    df_plt = pd.DataFrame({'scores': scores,
                        'samples': nr_samples, 
                            "class_ratio":class_ratio if len(class_ratio) > 0 else np.zeros(len(scores))})

    # plot RMSE over dataset types
    fig, ax1 = plt.subplots(figsize=(15, 10))
    df_plt['samples'].plot(kind='line', marker='d', secondary_y=True, ylabel="# Samples").set_ylabel("# Samples")
    df_plt['scores'].plot(kind='bar', color='r', ylabel="F1 score" if params["predict"]=="class" else "RMSE").set_xticklabels(classifiers) 
    if len(class_ratio)>0:
        df_plt['class_ratio'].plot(kind='bar', color='orange')
    plt.xlabel("Classifiers")
    ax1.legend(["Classification", "Class Ratios"])

    plt.title("Comparing "+("F1 " if params["predict"]=="class" else "RMSE ")+"of different classifiers")
    plt.savefig(OUTPUT_DIR+"rmse_comparison.png",bbox_inches = "tight")
    plt.clf()
    #plt.show()
    #print(test_scores)

    print(f'Average : {"f1 " if params["predict"]=="class" else "RMSE "}{np.mean(scores)}')

    test_scores_lst = list(test_scores.items())

    srted = sorted(test_scores_lst, key=lambda tup: tup[1])
    #srted = test_scores_lst.sort(key=lambda x:x[1])
    print("Sorted:")
    for c,s in srted:
        print("   ", c, '->', s)
        

    # For each feature generate a list of all indices where it appears over various classifiers    
    
    overal_top_feat = {}
    
    #dict with each feature name : [importance index list]
    #iterate over top_n_features[clf] (per classifier)
    for clf in top_n_features.keys():
        #iterate over top features zipped list
        top_feat_lst = top_n_features[clf]
        for k in range(len(top_feat_lst)):
            cur_feat_name = top_feat_lst[k][0]
            cur_feat_importance = top_feat_lst[k][1]
            if cur_feat_name in overal_top_feat:
                overal_top_feat[cur_feat_name] =  overal_top_feat[cur_feat_name] + [k]
            else:
                overal_top_feat[cur_feat_name] = [k]

    # get overal ranking sum (the one with the smallest rank is the most important)
    clf_count = len(list(overal_top_feat.values())[0])
    for feat in overal_top_feat.keys():
        
        if len(overal_top_feat[feat]) < clf_count:
            dist = clf_count - len(overal_top_feat[feat])
            
            overal_top_feat[feat] =  overal_top_feat[feat] + ([50]*dist)
        overal_top_feat[feat] = sum(overal_top_feat[feat])/clf_count +1
        
    overal_top_feat = dict(sorted(overal_top_feat.items(), key=lambda item: item[1]))

    top_n_features["Overal most important"] = list(overal_top_feat.keys())
    top_n_features["Avg rank"] = list(overal_top_feat.values())

    today = date.today()
    output = today.strftime("%d_%m_%Y")
    top_n_features_pd = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in top_n_features.items() ]))
    top_n_features_pd.to_excel(OUTPUT_DIR+output+".xlsx")
    
    # correlation
    #min = len(y_lst[0])
    #for i in range(len(y_lst)):
    #    ln = len(y_lst[i])
    #    if ln < min:
    #        min = ln
    #    
    #y1 = y_lst[0][:min-1]
    #y2 = y_lst[1][:min-1]
    #print(f"Correlation {stats.pearsonr(y1,y2)}")

if __name__ == "__main__":
    main(sys.argv)