
import logging as lg
import textwrap
import time
import math
import coloredlogs
import constants as CS
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from pandas.core.frame import DataFrame
from PIL import Image, ImageDraw, ImageFont

coloredlogs.install()
NR_APPENDED_COLS = 2

def id_text_at_end(df):
    # move post_id and post_text cols to end
    col_list = list(df.columns)
    if CS.LOAD_LIWC:
        col_list.remove(CS.LIWC_PREFIX+"post_id")
    if CS.LOAD_FOUNDATIONS:
        col_list.remove(CS.FOUNDATIONS_PREFIX+"post_id")

    col_list.remove("post_id")
    col_list.remove("post_text")
    col_list.append("post_id")
    col_list.append("post_text")
    
    df = df[col_list]
    return df

def drop_non_feature_cols(df):
     # Drop all non feature columns
    foundation_prefix = [CS.FOUNDATIONS_PREFIX] if CS.LOAD_FOUNDATIONS else []
    liwc_prefix = [CS.LIWC_PREFIX] if CS.LOAD_LIWC else []
    prefixes = set(list(CS.FEATURES_TO_GENERATE_MONO.keys()) +list(CS.FEATURES_TO_GENERATE_MP.keys())+foundation_prefix+liwc_prefix)

    for c in df.columns:
        flag = len([pref for pref in prefixes if pref in c]) > 0 and not "post_id" in c
        if not flag:
            lg.info("      not visualising "+c)
            df = df.drop(c, axis=1)
    return df

def df_get_text_samples(df):
    """ Get the sample texts of every feature to be displayed in png

        returns : [[(post_id, feature value), (post_id, feature value), (post_id, feature value)],...], 

        Parameters
        ----------
        df : Dataframe
            dataframe with potentialy multiple columns    
    """
    RANGE = 0.1
    SAMPLE_FRAC = 0.5
    
    lg.info("  Drawing samples text with sample frac of "+str(SAMPLE_FRAC))
   
    df = id_text_at_end(df)
    df_sampeled = df.sample(frac=SAMPLE_FRAC)

    """ Define low, mid and top ranges to display posts for for each column
    s.t. we only need to iterate once over entire sampled DataFrame """


    range_list = [] #[[low_range, mid_range, top_range],...], each range is a list of length 2
    nr_feature_columns = len(df.columns)-NR_APPENDED_COLS

    for i in range(nr_feature_columns):
        max = df_sampeled.iloc[:, i].max()

        min = df_sampeled.iloc[:, i].min()
        length = max * RANGE
        low_range = [int(min), int(min+length)]
        mid_range = [int(max/2-length/2), int(max/2+length/2)]
        top_range = [int(max-length), int(max)]
        
        
        range_list.append([low_range, mid_range, top_range])

    examples_list = [[False, False, False] for _ in range(nr_feature_columns)]

    for i, row in enumerate(df_sampeled.itertuples(), 1):
        for j in range(nr_feature_columns):
            value = row[j+1]
        
            #check if value is in any ranges (low, mid, top)
            for cur_range_idx in range(len(range_list[j])):

                cur_range_lower = (range_list[j][cur_range_idx][0]) 
                cur_range_upper = (range_list[j][cur_range_idx][1])
                
                if value in range(cur_range_lower, cur_range_upper+1):
                    if not examples_list[j][cur_range_idx]:                       
                        examples_list[j][cur_range_idx] = (row[len(row)-NR_APPENDED_COLS], value)

    return examples_list, [ top_range, mid_range, low_range]

def df_to_text_png(df):
    """Generate png that shows 3 post example of each df column 

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """
    lg.info("  drawing post examples")
    df_to_text_png_timer = time.time()
    
    
    ex_list, ranges_min_max = df_get_text_samples(df)

    MAX_NR_LINES = 20
    MAX_CHARCTER_PER_LINE = 70
    EXAMPLE_OFFSET_X,EXAMPLE_OFFSET_Y = 10, 50
    Y_GAP = 40

    WIDTH = 400
    HEIGHT = 100+MAX_NR_LINES*3*10+EXAMPLE_OFFSET_Y*3

    font = ImageFont.truetype(CS.HOME_DIR+"misc/bins/DejaVuSans.ttf")
    
    nr_cols = min((len(df.columns)-NR_APPENDED_COLS),CS.NR_COLS_TEXT)
    nr_rows = math.ceil((len(df.columns)-NR_APPENDED_COLS)/nr_cols)
    W, H = (WIDTH*nr_cols,HEIGHT*nr_rows)
    img = Image.new("RGBA",(W,H),"white")
    draw = ImageDraw.Draw(img)


    for i in range(len(ex_list)):
        for j in range(len(ex_list[i])):

            # if sample is not set we write empty string
            if not ex_list[i][j]:
                ex_list[i][j] = "N/A"
            else:
                value = ex_list[i][j][1]
                post_id = ex_list[i][j][0]
                post_text = df.loc[df.post_id == post_id,'post_text'].values[0]
                ex_list[i][j] = "{0} ({1}): {2}".format(post_id, value, post_text)

    
    horizontal_gap = 0
    
    for cat_index in range(len(ex_list)):
        cat_title = df.columns[cat_index]
        vertical_gap = 0

        #draw cateogry title (i.e. df.column name)
        row_offset = math.floor(cat_index/CS.NR_COLS_TEXT)*HEIGHT
        w, h = draw.textsize(cat_title)
        draw.text(((WIDTH*cat_index+(WIDTH-w)/2)% W,20+row_offset), cat_title,  fill="black", font=font)
        
        
        for ex_index in reversed(range(len(ex_list[cat_index]))):    
            #draw range type (i.e. top range, mid range, low range)
            rang_name = [ s + " range:" for s in ["low", "mid", "top"]][ex_index]

            #ranges_min_max_reversed  = ranges_min_max.reverse()
            #rang_name += " "+str(ranges_min_max_reversed[ex_index]) #TODO: net yet working

            w_r, h_r = draw.textsize(rang_name)
            draw.text(((WIDTH*cat_index+(WIDTH-w_r)/2)%W,EXAMPLE_OFFSET_Y+vertical_gap+row_offset), rang_name,  fill="black")

            #draw each example below each other
            text = ex_list[cat_index][ex_index]

            lines = textwrap.wrap(text, width=MAX_CHARCTER_PER_LINE)
            y_text = h
            for line_index in range(min(len(lines),MAX_NR_LINES)):
                line = lines[line_index]
                
                width, height = 8,10
                x = ((WIDTH*cat_index+(WIDTH-w)/2)-100) % W
                y = EXAMPLE_OFFSET_Y+y_text+vertical_gap+row_offset
                
                draw.text((x, y), line, fill="black", font=font)
                y_text += height

            # After first block, add height of previous block and Y_GAP contstant as gap between blocks 
            vertical_gap+= h*MAX_NR_LINES+Y_GAP
        horizontal_gap += EXAMPLE_OFFSET_X+MAX_CHARCTER_PER_LINE*6

    #lg.info("    DURATION: {0} ".format(round(time.time() - df_to_text_png_timer,2))) 
    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    img.save("{0}text_samples{1}.png".format(CS.OUTPUT_DIR, mini))


def df_to_plots(df_features):
    """Generate matplotlibs for each df column and save directly as report.png

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """
    lg.info("  drawing plot")
    
    df_to_plots_timer = time.time()
    df_features = drop_non_feature_cols(df_features)

    nr_cols =  CS.NR_COLS_MPL#len(list(df_features.columns))%6
    nr_rows = len(list(df_features.columns))//5+1

    plt.rcParams["figure.figsize"] = (4*nr_cols,4*nr_rows)

    fig, axs = plt.subplots(nr_rows,nr_cols, sharex=False, sharey=False)
    index_sum = 0
    for i in range(nr_rows):
            for j in range(nr_cols):
                #terminate if no more features available
                if index_sum >= len(list(df_features.columns)):
                    break

                data = df_features.iloc[:,index_sum].to_list()
                nr_bins = 300 #TODO: set correct amount of bins
                #nr_bins = abs(np.max(data)-np.min(data))

                if axs.ndim > 1:
                   
                    axs[i, j].hist(data,bins=nr_bins, align="mid") 
                    axs[i, j].set_title(df_features.columns[index_sum])                    
                else:
                   
                    axs[j].hist(data, bins=nr_bins, align="mid") 
                    axs[j].set_title(df_features.columns[index_sum])
                index_sum +=1

            #terminate if no more features available        
            if index_sum >= len(list(df_features.columns)):
                    break

    
    
    mini_text = "Using minified data ({0} fraction)".format(CS.MINIFY_FRAC) if CS.USE_MINIFIED_DATA else "Using complete dataset"
    plt.suptitle(mini_text, fontsize=16)
    if df_features.shape[1] < 25:
        plt.show()

    mini = "_mini" if CS.USE_MINIFIED_DATA else ""
    fig.savefig("{0}graphs{1}.png".format(CS.OUTPUT_DIR, mini))
    #lg.info("    DURATION: {0} ".format(round(time.time() - df_to_plots_timer,2)))


def generate_report(df):
    """Generate report by visualising the columns of the dataframe as 
        histograms and listing 3 example post (low, medium, high value)
        below the histograms. Saves as report.png in home directory of script

    Parameters
    ----------
    df : Dataframe
        Dataframe with format of feature_to_df

    """
    lg.info("Generating report")
    df_to_plots(df)
    df_to_text_png(df)
