import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tune_sklearn import TuneGridSearchCV
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn import metrics
import json
import multiprocessing
import itertools as it
from tqdm import tqdm

DO_SHAPLY = True
OUTPUT_DIR = "./output/"
DATASETS_DIR = "./datasets/"


def get_data(params):
    prepend_csv = "prepend_mf_liwc_angel_info_topic_scores_reactions_reduced_da.csv"
    standalone_csv = "standalone_liwc_mf_angel_info_topic_scores_reduced_reactions_da.csv"

    if params["title_prepend"]:
        df = load_wo_cols(DATASETS_DIR+prepend_csv, params)
    else:
        df = load_wo_cols(DATASETS_DIR+standalone_csv, params)

    if params["new_reactions"]:
        new_react = "id_to_reactions_new.csv"
        df_reactions = pd.read_csv(DATASETS_DIR+new_react)
        df = df.merge(df_reactions, left_on="post_id", right_on="post_id",
                      validate="1:1", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    if params["norm"] < 2:
        df = df[df.columns.drop(
            list(df.filter(regex="_abs" if params["norm"] == 1 else "_norm")))]

    keys = ["info", "yta", "nah", "esh", "nta"]
    weight = "weighted_" if params["weighted"] else ""
    values = ["reactions_"+weight+k.upper() for k in keys]
    acros = dict(zip(keys, values))

    dfs = []
    if params["topics_separate"] > 0:

        topic_min = df["topic_nr"].min()
        topic_max = df["topic_nr"].max()
        #print(f"Data split by topic ({topic_min}, {topic_max})")

        for i in range(topic_min, topic_max+1):
            dfs.append(df.loc[df["topic_nr"] == i])
    else:
        dfs = [df]

    #print(f"Number of dataframes: {len(dfs)}")

    return dfs, acros


def load_wo_cols(path, params, remove_cols=[], verbose=False):
    cols_to_remove = ["post_text", "Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "Unnamed: 0.1",
                      "Unnamed: 0.1.1", "liwc_post_id", "foundations_post_id",
                      "foundations_title_post_id", "liwc_title_post_id", "post_created_utc"]+remove_cols
    metadata = ["speaker_account_comment_karma", "post_num_comments", "speaker_account_age",
                "speaker_account_link_karma", "post_ups", "post_downs", "post_score", "reactions_is_devil", "reactions_is_angel"]
    # removed "post_ratio" from metadata b.c. used for weights

    removed = []
    df = pd.read_csv(path, nrows=10)
    cols_to_read = list(df.columns)

    # remove metadata
    if params["wo_metadata"]:
        cols_to_remove = cols_to_remove+metadata

    # replace reactions with new ones TODO: why are they different?
    if params["new_reactions"]:
        cols_to_remove = cols_to_remove + \
            list(filter(lambda x: "reaction" in x and not "reaction_is" in x, cols_to_read))

    # remove liwc
    if not params["use_liwc"]:
        cols_to_remove = cols_to_remove + \
            list(filter(lambda x: "liwc_" in x, cols_to_read))

    # remove moral foundations
    if not params["use_mf"]:
        cols_to_remove = cols_to_remove + \
            list(filter(lambda x: "foundations_" in x, cols_to_read))

    # post requirements setup
    cols_to_remove = [
        x for x in cols_to_remove if x not in list(params["requirements"].keys())]

    if verbose:
        print(cols_to_read)
    for col in cols_to_remove:
        if col in cols_to_read:
            cols_to_read.remove(col)
            removed.append(col)

    #print(f"Removed {removed} from {path.split('/')[-1]}")
    df = pd.read_csv(path, usecols=cols_to_read, nrows=100000)

    # delte posts that don't meet requirements
    nr_rows_pre_req = len(df)
    for k, v in params["requirements"].items():
        df = df.loc[(df[k] >= v), :]
    # print(
    #    f"Removed {int(100*(nr_rows_pre_req-len(df))/len(df))}% due to requirements, Now {len(df)} posts remain.")
    # Check values in df
    # df.describe().loc[['min','max']].to_csv("min_max.csv",index=False)
    return df


def sampling(X_train, y_train, params, indices=[], verbose=False):
    df_len_old = len(X_train)
    if verbose:
        print(f"{params['sampling']}-sampling for {params['predict']}")

    if params["sampling"] == "none":
        X_train_ret = X_train
        y_train_ret = y_train

    if verbose:
        print("Original Y distribution on training set")
        _ = plt.hist(y_train, bins='auto')
        plt.show()

    if params["predict"] == "ratio":
        if params["sampling"] == "up":
            raise Exception("Upsampling with regression is not feasible☹️")
        elif params["sampling"] == "down":
            # downsampling
            bucket_ranges = [x/10 for x in list(range(0, 11))]
            bucket_counter = []

            X_train_tmp = X_train
            y_train_tmp = y_train.reshape((len(y_train), 1))
            dummy_feat_name = [str(int) for int in range(X_train_tmp.shape[1])]
            feat_names_to_sample = dummy_feat_name+["Y"]
            data_to_sample = np.append(X_train_tmp, y_train_tmp, 1)
            df_to_sample = pd.DataFrame(
                data_to_sample, columns=feat_names_to_sample)

            # Get bucket sizes
            for i in range(len(bucket_ranges)):
                if bucket_ranges[i] == 1:
                    continue
                orig_size = len(df_to_sample.loc[(bucket_ranges[i] <= df_to_sample['Y']) & (
                    df_to_sample['Y'] <= bucket_ranges[i+1])])
                bucket_counter.append(orig_size)

            # We only downsample buckets that are > 2* bucket mean => 2*bucket mean
            bucket_max = int(np.mean(bucket_counter)*1.5)
            for j in range(len(bucket_counter)):
                if bucket_counter[j] > bucket_max:
                    if verbose:
                        print(
                            f"Bucket {bucket_ranges[j]}-{bucket_ranges[j+1]} has {bucket_counter[j]}>{bucket_max}")
                    df_bkt = df_to_sample.loc[(bucket_ranges[j] <= df_to_sample['Y']) & (
                        df_to_sample['Y'] <= bucket_ranges[j+1])]
                    df_bkt_smpl = df_bkt.sample(
                        n=int(bucket_max), replace=False, random_state=42)
                    df_to_sample.loc[(bucket_ranges[j] <= df_to_sample['Y']) & (
                        df_to_sample['Y'] <= bucket_ranges[j+1])] = df_bkt_smpl

            df_to_sample = df_to_sample.dropna()
            y_train = df_to_sample["Y"]
            df_to_sample = df_to_sample.drop(columns=["Y"])

            X_train = df_to_sample.to_numpy()
            X_train_ret = X_train
            y_train_ret = y_train

    elif params["predict"] == "class":
        df_y = pd.DataFrame(data={"Y": y_train})

        if len(indices) > 0:
            if verbose:
                print(f"Using {len(indices)} indices")
        else:
            indices = range(len(indices))

        # Get list of indices for classes that are in the indices array
        c0_idx = pd.Series(df_y.loc[df_y["Y"] == 0].index.values)
        c0_idx = c0_idx[c0_idx.isin(indices)]
        c1_idx = pd.Series(df_y.loc[df_y["Y"] == 1].index.values)
        c1_idx = c1_idx[c1_idx.isin(indices)]

        if verbose:
            print(f"    Y=0: {c0_idx.shape}")
            print(f"    Y=1: {c1_idx.shape}")

        if params["sampling"] == "up":
            # upsample
            if len(c0_idx) >= len(c1_idx):
                n = len(c0_idx)
                c1_idx_sampeled = c1_idx.sample(
                    n=n, random_state=1, replace=len(c1_idx) < n).values
                c0_idx_sampeled = c0_idx.values
                if verbose:
                    print(f"Upsampling Y=1 with {n} samples")

            elif len(c0_idx) < len(c1_idx):
                n = len(c1_idx)
                c0_idx_sampeled = c0_idx.sample(
                    n=n, random_state=1, replace=len(c0_idx) < n).values
                c1_idx_sampeled = c1_idx.values
                if verbose:
                    print(f"Upsampling Y=0 with {n} samples")

        elif params["sampling"] == "down":
            # downsample
            if len(c0_idx) >= len(c1_idx):
                n = len(c1_idx)
                c0_idx_sampeled = c0_idx.sample(
                    n=n, random_state=1, replace=len(c0_idx) < n).values
                c1_idx_sampeled = c1_idx.values
                if verbose:
                    print(f"Downsampling Y=0 with {n} samples")
            elif len(c0_idx) < len(c1_idx):
                n = len(c0_idx)
                c1_idx_sampeled = c1_idx.sample(
                    n=n, random_state=1, replace=len(c1_idx) < n).values
                c0_idx_sampeled = c0_idx.values
                if verbose:
                    print(f"Downsampling Y=1 with {n} samples")
        else:
            c0_idx_sampeled = c0_idx
            c1_idx_sampeled = c1_idx

        all_idx = np.concatenate((c0_idx_sampeled, c1_idx_sampeled), axis=0)

        if verbose:
            df_tmp = df_y.iloc[all_idx]
            print(f"   Y=0: {len(df_tmp.loc[df_tmp['Y']==0])}")
            print(f"   Y=1: {len(df_tmp.loc[df_tmp['Y']==1])}")

        X_train_ret = X_train[all_idx, :]
        y_train_ret = y_train[all_idx]

    # print(df_len_old)
    #print(f"Removed/Added {int(100*(df_len_old-len(y_train_ret))/len(y_train_ret))}% due to Sampling, Now {len(y_train_ret)} posts remain.")
    return X_train_ret, y_train_ret


def obj_size_fmt(num):
    if num < 10**3:
        return "{:.2f}{}".format(num, "B")
    elif ((num >= 10**3) & (num < 10**6)):
        return "{:.2f}{}".format(num/(1.024*10**3), "KB")
    elif ((num >= 10**6) & (num < 10**9)):
        return "{:.2f}{}".format(num/(1.024*10**6), "MB")
    else:
        return "{:.2f}{}".format(num/(1.024*10**9), "GB")


def memory_usage():
    memory_usage_by_variable = pd.DataFrame({k: sys.getsizeof(v)
                                             for (k, v) in globals().items()}, index=['Size'])
    memory_usage_by_variable = memory_usage_by_variable.T
    memory_usage_by_variable = memory_usage_by_variable.sort_values(
        by='Size', ascending=False).head(10)
    memory_usage_by_variable['Size'] = memory_usage_by_variable['Size'].apply(
        lambda x: obj_size_fmt(x))
    return memory_usage_by_variable


def opposite_jdgmt(judg):

    if "NTA" in judg:
        rtn = judg.replace("NTA", "YTA")
    elif "NAH" in judg:
        rtn = judg.replace("NAH", "ESH")
    elif "YTA" in judg:
        rtn = judg.replace("YTA", "NTA")
    elif "ESH" in judg:
        rtn = judg.replace("ESH", "NAH")
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

    if mapping == "opposite" or mapping == "map":
        #print("Map = opposite")
        for k in acros.keys():
            acr = acros[k]

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
            # set negative values to 0 & add opposite judgement vote
            df[acr] = df[acr] + df[opposite_jdgmt(acr)]

    elif mapping == "clip":
        #print("Map = clip")
        for k in acros.keys():
            acr = acros[k]
            df.loc[df[acr] < 0, acr] = 0

    return df


def get_data_classes(df, acros, ratio=0.5, verbose=False, predict="class", judgement_weighted=True, mapping="clip"):
    if verbose:
        print(f"df original shape {df.shape}")

    n_rows_old = len(df)

    # Map negative judgements to opposing judgement, if we are not simply counting each comment as one vote (i.e. if judgement_weighted = True)
    # i.e. YTA<->NTA, ESH<->NAH
    if judgement_weighted:
        df = map_negative_values(df, acros, mapping=mapping)

    if predict == "class":
        # We only look at YTA and NTA
        df["YTA_ratio"] = df[acros["yta"]] / \
            (df[acros["info"]] + df[acros["yta"]] +
             df[acros["nah"]]+df[acros["esh"]]+df[acros["nta"]])

        # drop all rows where the majority is not YTA or NTA
        df = df.loc[((df[acros["yta"]] > df[acros["info"]]) & (df[acros["yta"]] > df[acros["nah"]]) & (df[acros["yta"]] > df[acros["esh"]])) | (
            (df[acros["nta"]] > df[acros["info"]]) & (df[acros["nta"]] > df[acros["nah"]]) & (df["reactions_weighted_NTA"] > df[acros["esh"]]))]
        if verbose:
            print(f"Drop all rows where majority is not YTA or NTA {df.shape}")

        # drop all rows that are not "extreme" enough
        df = df.loc[(1-ratio <= df["YTA_ratio"]) | (df["YTA_ratio"] <= ratio)]

        #print(
        #    f"Removed {int(100*( (n_rows_old-len(df)) / n_rows_old) )}% due to agreement ratio, Now {len(df)} posts remain.")

        # specifc classes & drop unnecesarry
        # YTA = Class 1, NTA = class 0
        df["Y"] = np.where(df[acros["yta"]] > df[acros["nta"]], 1,  0)
        smp_weights = df["post_ratio"]
        if verbose:
            print(df.shape)

    elif predict == "ratio":
        # Y = asshole ratio(AHR) = (YTA+ESH)/(YTA+ESH+NTA+NAH)
        # drop posts w.o. votes
        tmp = df[acros["yta"]] + df[acros["nah"]] + \
            df[acros["esh"]]+df[acros["nta"]]
        tmp = tmp[tmp != 0]
        tmp = (df[acros["yta"]]+df[acros["esh"]])/tmp
        df["Y"] = tmp

        n_rows_old = len(df)
        df = df.loc[(1-ratio <= df["Y"]) | (df["Y"] <= ratio)]
        smp_weights = df["post_ratio"]
        # print(
        #    f"Removed {int(100*(n_rows_old-len(df))/len(df))}% of posts b.c. not enough agreement. Now {df.shape}")

    if np.min(df["Y"]) < 0 or np.max(df["Y"]) > 1:
        raise Exception("Y value should be in range [0,1]")

    # get list of all columns that contain uppercase vote acronym
    vote_acroynms = list(filter(lambda x: any(
        [acr.upper() in x for acr in list(acros.keys())]), list(df.columns)))
    vote_acroynms += ["post_id"]
    df = df.drop(columns=vote_acroynms)

    if verbose:
        print(df.shape)

    X = df.drop(columns=["Y"])
    y = df["Y"].to_numpy()

    feat_name_lst = list(X.columns)

    # scaling
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, feat_name_lst, smp_weights.to_numpy()


def get_train_test_split(params, grid_search=False, verbose=False):
    dfs, acros = get_data(params)

    df = dfs[0]
    if len(dfs) > 1:
        print("MORE THAN 1 df")

    df_cpy = df.copy()
    X, y, feat_name_lst,smp_weights = get_data_classes(df_cpy, ratio=params["ratio"], acros=acros, predict=params["predict"], judgement_weighted=params["weighted"],
                                           mapping=params["mapping"], verbose=False)
    if grid_search:
        print("YOU SURE YOU WANT TO BE DOING THIS?")
        return X, y, feat_name_lst

    train, test = train_test_split(
        range(len(X)), test_size=0.33, random_state=42)

    X_train, y_train = sampling(
        X[train], y[train], params, indices=train if params["predict"] == "class" else [], verbose=False)

    X_test = X[test, :]
    y_test = y[test]

    if params["random_y"]:
        # Sanity check, i.e. get results for random predition
        #df["Y"] = np.random.randint(0, 1001, size=len(df["Y"]))/1000
        y_test_sum_old = np.sum(y_test[:1000])
        np.random.shuffle(y_test)
        y_test_sum_new = np.sum(y_test[:1000])
        if y_test_sum_old == y_test_sum_new:
            raise Exception("Not truly random values")
        if verbose:
            print(f"USING RANDOM Y\n Was {y_test_sum_old} Is {y_test_sum_new}")

    return X_train, y_train, X_test, y_test, feat_name_lst


def get_clf_name(params, clf_type):
    clf_name = clf_type
    for k, v in params.items():
        if isinstance(v, bool) and v:
            clf_name += f"_{k}"
        else:
            clf_name += f"_{k}={v}"
    return clf_name


def get_metrics(y_test, y_pred, params, verbose=True):
    if params["predict"] == "class":
        # testing score
        f1_test = metrics.f1_score(y_test, y_pred, average="weighted")
        acc_test = metrics.accuracy_score(y_test, y_pred)

        if verbose:
            print(f"    Accuracy: {acc_test}\n    F1: {f1_test}")
            print(classification_report(y_test, y_pred, target_names=[
                "Class 0: low AH", "Class 1: high AH"]))
        else:
            return f1_test

    elif params["predict"] == "ratio":
        mean_abs = metrics.mean_absolute_error(y_test, y_pred)
        mean_sqr = metrics.mean_squared_error(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

        if verbose:
            print(
                f"    Mean absolute: {mean_abs}\n    Mean squared: {mean_sqr}\n    Root Mean Squared: {rmse}")
        else:
            return mean_abs


def get_param_combs(params,):
    combinations = list(it.product(*(params[Name] for Name in params)))

    keys = list(params.keys())
    combs = list(map(lambda x: dict(zip(keys, x)), combinations))

    return combs


def main():
    params = {
        # normalised: 0 = only "abs", 1 = only "norm", 2 = norm and abs
        "norm": [0,1,2],
        # weighted_vals: whether votes should be weighted by comment score
        "weighted": [True, False],
        # title_prepend: whether to use the title prepended or standalone dataset
        "title_prepend": [True,False ],
        # sampling_vals: which type of sampling should be done ("up", "down", "none")
        "sampling": ["up", "down", "none"],
        # if each topic should be analysed separately
        "topics_separate": [False, True],
        # should we predict "class" (classification for binary) or "ratio" (regression for AHR)
        "predict": ["class","ratio", ],
        # should we "clip" negative votes or map them to the "opposite"
        "mapping": ["opposite", "clip"],
        # which most extreme AHR or YTA_ratio we want to predict 0.3, 0.2, 0.1, 0.05
        "ratio": [0.5,0.3, 0.2, 0.1, 0.05],
        # wheter we should include metadata columns (e.g. post_score, account_karam, link_karma) set MANUALLY
        "wo_metadata": [True, False],
        # wheter we should use the old or new reactions (reactions_YTA, NTA)
        "new_reactions": [False],
        "use_liwc": [True],  # wheter we use liwc features
        "use_mf": [True],  # whether we use moral foundation features
        "requirements": [True, False],
    }

    post_requirements = {  # requirement: key >= value in post
        "post_num_comments": 10,
        "post_score": 10,
        "post_ratio": 0.7,
    }

    models_to_compare = []
    # wheter we a random run right now => to compare the actual score with the random one
    random_run = [True, False]

    combs = get_param_combs(params)
    for params_i in tqdm(combs):

        # upsamping not implemented for regression
        if params_i["sampling"] == "up" and params_i["predict"] == "ratio":
            continue

        # handle post requirements
        if params_i["requirements"]:
            params_i["requirements"] = post_requirements
        else:
            params_i["requirements"] = dict.fromkeys(post_requirements, 0)

        last_random_score = None  # holder variable for last random score
        for is_random in random_run:
            params_i["random_y"] = is_random

            # ADD GPU
            #xgboost = xgb.XGBClassifier(verbosity=0, random_state=42, use_label_encoder=False, tree_method='gpu_hist') if params_i["predict"] == "class" else xgb.XGBRegressor(
            #    verbosity=0, random_state=42, tree_method='gpu_hist')

            xgboost = xgb.XGBClassifier(verbosity=0, random_state=42, use_label_encoder=False) if params_i["predict"] == "class" else xgb.XGBRegressor(
                verbosity=0, random_state=42)
            classifiers = (xgboost, "xgboost")
            clf_name = get_clf_name(params_i, classifiers[1])
            X_train, y_train, X_test, y_test, feat_name_lst = get_train_test_split(
                params_i)

            smp_weights = None
            xgboost.fit(X_train, y_train, sample_weight=smp_weights)
            y_pred = xgboost.predict(X_test)

            is_regression = params_i["predict"] == "ratio"

            nr_samples = X_train.shape[0]
            nr_features = X_train.shape[1]
            complexity = nr_samples*nr_features
            score = get_metrics(y_test, y_pred, params_i, verbose=False)

            if is_random:
                last_random_score = score
                improvement_rnd = -42
            else:
                # how much better the actual run was compared to the random
                improvement_rnd = score - \
                    last_random_score if params_i["predict"] == "class" else last_random_score-score

                print(f'{"F1" if params_i["predict"] == "class" else "ME" }: {score}, Improvement to Shuffle: {improvement_rnd}')

            models_to_compare.append(
                [clf_name, score, improvement_rnd, complexity, nr_samples, nr_features, is_random, is_regression])

    models_df = pd.DataFrame(models_to_compare, columns=[
                             "Name", "Score", "Improvement", "Complexity", "Nr Samples", "Nr Features", "is_random", "is_regression"])
    models_df.to_csv(OUTPUT_DIR+"model_comparisons.csv", index=False)

if __name__ == "__main__":
    main()
