from re import T

from helpers.helper_functions import *
from bertopic import BERTopic
import constants as CS
import pandas as pd 
import logging as lg
import coloredlogs
from matplotlib import pyplot as plt

coloredlogs.install()

def topic_modelling(posts_raw, post_ids):
    """ Generate topics of all posts, return dataframe with each posts's topic id, and its probabilty. 
        Addtionnaly, save a dataframe in output directory that shows the mapping of each topic id to each topic string
        Lastly, save a visualisation of the topics

    Args:
        posts_raw (list): list of all raw_post texts
        post_ids (list): list of 

    Returns:
        topic_prob_df: dataframe with the columns "post_id", "topic_nr", "topic_probability"
    """
    lg.info("Generating topics")
    
    # lemmatize?
    post_list_clean = [get_clean_text(post,
                            globals_loader.nlp,
                            remove_punctuation=False,
                            remove_stopwords=True,
                            do_lemmatization=False) 
                        for post in posts_raw]

    # min topic size?
    model = BERTopic(language="english", min_topic_size=int(CS.MIN_CLUSTER_PERC*len(post_ids)), nr_topics="auto", low_memory=True, calculate_probabilities=False)
    topics, _ = model.fit_transform(post_list_clean)

    actual_nr_topics = len(set(topics))
    desired_nr_topics = int(CS.TOPICS_ABS)
    #if actual_nr_topics > int(CS.TOPICS_ABS):
    #    topics, probs = model.reduce_topics(post_list_clean, topics, probabilities=probs, nr_topics=desired_nr_topics)

    # get dataframe to return
    info_df = model.get_topic_info()
    topic_prob_df = pd.DataFrame(
        data = list(zip(post_ids, topics)), #data = list(zip(post_ids, topics, probs)),
        columns = ["post_id", "topic_nr", "topic_probability"])

    print(info_df.head(3))

    topic_ids = info_df["Topic"]
    topic_sizes = info_df["Count"]

    # chart of size per topic
    size_per_topic_dir = "{0}{1}_{2}{3}.png".format(CS.OUTPUT_DIR, "size_per_topic", get_date_str(),  "_mini" if CS.USE_MINIFIED_DATA else "")
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    
    ax.bar(x=topic_ids, height = topic_sizes)
    invalid_nr = info_df.loc[0,"Count"]
    total_nr = info_df["Count"].sum()
    title = "Nr valid posts: {0}, Nr invalid posts:{1}".format(total_nr-invalid_nr, invalid_nr)
    ax.set_title(title)
    fig.savefig(size_per_topic_dir)

    # Save dataframe with mapping from topic id to topic string
    MAX_WORDS_PER_TOPIC = 4
    topic_strings = list(map(lambda topic_nr: "_".join([tpl[0] for tpl in model.get_topic(topic_nr)[:MAX_WORDS_PER_TOPIC]]), topics))

    topic_nr_to_name_df = pd.DataFrame(
        data = list(zip(topics, topic_strings)),
        columns = ["topic_nr", "topic_str"])
    
    mapping_dir = "{0}{1}_{2}{3}.csv".format(CS.OUTPUT_DIR, "topic_nr_to_str", get_date_str(), "_mini" if CS.USE_MINIFIED_DATA else "")
    topic_nr_to_name_df.to_csv(mapping_dir)

    # Save topic visualisation
    visualsation_dir = "{0}{1}_{2}{3}.html".format(CS.OUTPUT_DIR, "topic_visualsation", get_date_str(),  "_mini" if CS.USE_MINIFIED_DATA else "")
    fig = model.visualize_topics()
    fig.write_html(visualsation_dir)

    visualsation_dir_bar = "{0}{1}_{2}{3}.html".format(CS.OUTPUT_DIR, "topic_barchchart", get_date_str(),  "_mini" if CS.USE_MINIFIED_DATA else "")
    fig_bar = model.visualize_barchart()
    fig_bar.write_html(visualsation_dir_bar)

    visualsation_dir_hier  = "{0}{1}_{2}{3}.html".format(CS.OUTPUT_DIR, "topic_hierarchy", get_date_str(),  "_mini" if CS.USE_MINIFIED_DATA else "")
    fig_hier = model.visualize_hierarchy()
    fig_hier.write_html(visualsation_dir_hier)


    lg.info("    generated {0} topics".format(len(set(topics))))

    # show representative sample of topic id 47
    # model.get_representative_docs(47)
    return topic_prob_df
    
