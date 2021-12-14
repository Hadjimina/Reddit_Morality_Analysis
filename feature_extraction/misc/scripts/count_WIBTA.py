import constants as CS
import helpers.globals_loader as globals_loader

globals_loader.init()  
df_posts = globals_loader.df_posts
df_post_text = df_posts["post_text"].to_list()
df_post_title = df_posts["post_title"].to_list()
wibta_posts = [txt for txt in df_post_text if "wibta" in txt.lower()] 
wibta_titles = [txt for txt in df_post_title if "wibta" in txt.lower()] 

print("{0} posts contain the word 'wibta' => {1}%".format(len(wibta_posts), round(len(wibta_posts)*100/len(df_post_text))))
print("{0} titles contain the word 'wibta' => {1}%".format(len(wibta_titles), round(len(wibta_titles)*100/len(df_post_title))))
