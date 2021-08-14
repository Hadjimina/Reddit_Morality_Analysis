import logging as lg

import coloredlogs
import constants as CS
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

coloredlogs.install()
import textwrap
import time

from PIL import Image, ImageDraw, ImageFont

NR_APPENDED_COLS = 2

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

    return examples_list

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
    
    ex_list = df_get_text_samples(df)

    MAX_NR_LINES = 20
    MAX_CHARCTER_PER_LINE = 70
    EXAMPLE_OFFSET_X,EXAMPLE_OFFSET_Y = 10, 50
    Y_GAP = 40

    WIDTH = 400
    HEIGHT = 100+MAX_NR_LINES*3*10+EXAMPLE_OFFSET_Y*3

    font = ImageFont.truetype(CS.HOME_DIR+"misc/bins/DejaVuSans.ttf")
    
    W, H = (WIDTH*(len(df.columns)-NR_APPENDED_COLS),HEIGHT)
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
        w, h = draw.textsize(cat_title)
        draw.text((WIDTH*cat_index+(WIDTH-w)/2,20), cat_title,  fill="black", font=font)
        
        
        for ex_index in reversed(range(len(ex_list[cat_index]))):    
            #draw range type (i.e. top range, mid range, low range)
            rang_name = [ s + " range:" for s in ["low", "mid", "top"]][ex_index]
            w_r, h_r = draw.textsize(rang_name)
            draw.text((WIDTH*cat_index+(WIDTH-w_r)/2,EXAMPLE_OFFSET_Y+vertical_gap), rang_name,  fill="black")

            #draw each example below each other
            text = ex_list[cat_index][ex_index]

            #replace " ' " encoding
            #if u"\u2018" in text or u"\u2019" in text:
            #    text = text.replace(u"\u2018", "'").replace(u"\u2019", "'")

            lines = textwrap.wrap(text, width=MAX_CHARCTER_PER_LINE)
            y_text = h
            for line_index in range(min(len(lines),MAX_NR_LINES)):
                line = lines[line_index]
                
                width, height = 8,10
                x = (WIDTH*cat_index+(WIDTH-w)/2)-100
                y = EXAMPLE_OFFSET_Y+y_text+vertical_gap
                
                draw.text((x, y), line, fill="black", font=font)
                y_text += height

            # After first block, add height of previous block and Y_GAP contstant as gap between blocks 
            vertical_gap+= h*MAX_NR_LINES+Y_GAP
        horizontal_gap += EXAMPLE_OFFSET_X+MAX_CHARCTER_PER_LINE*6

    lg.info("    DURATION: {0} ".format(round(time.time() - df_to_text_png_timer,2))) 
    img.save(CS.OUTPUT_DIR+"text_samples.png")


def df_to_plots(df):
    """Generate matplotlibs for each df column and save directly as report.png

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """
    lg.info("  drawing plot")

    df_to_plots_timer = time.time()

    # Dropt text
    df_features = df.drop(CS.POST_PEND, axis =1)

    nr_cols =  5#len(list(df_features.columns))%6
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
                
                if axs.ndim > 1:
                   
                    axs[i, j].hist(data,bins=100, align="mid") 
                    axs[i, j].set_title(df_features.columns[index_sum])                    
                else:
                   
                    axs[j].hist(data, bins=100, align="mid") 
                    axs[j].set_title(df_features.columns[index_sum])
                index_sum +=1

            #terminate if no more features available        
            if index_sum >= len(list(df_features.columns)):
                    break

    plt.show()
    fig.savefig(CS.OUTPUT_DIR+"graphs.png")

    lg.info("    DURATION: {0} ".format(round(time.time() - df_to_plots_timer,2)))
    df_to_text_png(df)
