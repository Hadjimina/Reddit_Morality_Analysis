import pandas as pd
df_non_spacy_topics_angel = pd.read_csv("/mnt/g/My Drive/Msc/Thesis/Coding/dataset_output/non_spacy_topics_angel.csv")
df_spacy_mf_liwc = pd.read_csv("/mnt/g/My Drive/Msc/Thesis/Coding/dataset_output/spacy_liwc_mf.csv")

#feature_df = df_non_spacy_topics_angel.merge(
#                  df_spacy_mf_liwc, left_on="post_id", right_on="post_id", validate="1:1",suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
#feature_df.to_csv("/mnt/g/My Drive/Msc/Thesis/Coding/dataset_output/spacy_liwc_mf_non_spacy_angel_topics.csv")

#print(feature_df.shape)
print(df_non_spacy_topics_angel.shape)
print(f"#liwc {len(list(filter(lambda x: 'liwc' in x, list(df_non_spacy_topics_angel.columns))))}")
print(f"#mf {len(list(filter(lambda x: 'foundations' in x, list(df_non_spacy_topics_angel.columns))))}")
print(df_spacy_mf_liwc.shape)
print(f"#liwc {len(list(filter(lambda x: 'liwc' in x, list(df_spacy_mf_liwc.columns))))}")
print(f"#mf {len(list(filter(lambda x: 'foundations' in x, list(df_spacy_mf_liwc.columns))))}")

cols1 = set(list(df_non_spacy_topics_angel.columns))
cols2 = set(list(df_spacy_mf_liwc.columns))

print(f"appear in both {cols2.intersection(cols1)}")