import coloredlogs
from pandas.core.frame import DataFrame
import constants as CS
import matplotlib.pyplot as plt
import logging as lg
coloredlogs.install()
from PIL import Image, ImageDraw, ImageFont
import textwrap

def df_get_text_samples(df):
    """ Get the sample texts of every feature to be displayed in png

        returns : [[low_range_sample, mid_range_sample, top_range_sample],...], 

        Parameters
        ----------
        df : Dataframe
            dataframe with potentialy multiple columns    
    """
    RANGE = 0.1
    SAMPLE_FRAC = 1

    df_sampeled = df.sample(frac=SAMPLE_FRAC)

    """ Define low, mid and top ranges to display posts for for each column
    s.t. we only need to iterate once over entire sampled DataFrame """


    range_list = [] #[[low_range, mid_range, top_range],...], each range is a list of length 2
    nr_feature_columns = len(df.columns)-1
    for i in range(nr_feature_columns):
        max = df.iloc[:, i].max()
        min = df.iloc[:, i].min()
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
                        examples_list[j][cur_range_idx] = row[len(row)-1]

    return examples_list

def df_to_text_png(df):
    """Generate png that shows 3 post example of each df column 

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """

    df_features = df.drop("post_text", axis=1)
    ex_list = df_get_text_samples(df_features)

    MAX_NR_LINES = 20
    MAX_CHARCTER_PER_LINE = 60
    EXAMPLE_OFFSET_X,EXAMPLE_OFFSET_Y = 20, 50
    Y_GAP = 40

    WIDTH = 400
    HEIGHT = 100+MAX_NR_LINES*3*10+EXAMPLE_OFFSET_Y*3

    lg.info("  drawing post examples")
    W, H = (WIDTH*(len(df_features.columns)-1),HEIGHT)
    img = Image.new("RGBA",(W,H),"white")
    draw = ImageDraw.Draw(img)


    for i in range(len(ex_list)):
        for j in range(len(ex_list[i])):
            ex_list[i][j] = ex_list[i][j]+": "+df.loc[df.post_id == ex_list[i][j],'post_text'].values[0]

    #ex_list = [["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis arcu metus, tincidunt eu nisl a, aliquet vulputate libero. Pellentesque lobortis maximus tincidunt. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu ipsum tempor, fringilla dui eu, aliquam velit. Fusce euismod lacus lectus, ac pretium nulla tempus sed. Morbi molestie varius ornare. Morbi ullamcorper faucibus erat. In imperdiet commodo sapien, ac bibendum nibh semper id. Mauris gravida tortor aliquam, ullamcorper purus ac, tincidunt magna. Nulla fermentum finibus augue, quis varius leo iaculis vitae. Ut porta sodales sem nec tincidunt. Suspendisse auctor vel ex ut elementum. Vivamus sit amet ornare massa, a condimentum nibh.","Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis arcu metus, tincidunt eu nisl a, aliquet vulputate libero. Pellentesque lobortis maximus tincidunt. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu ipsum tempor, fringilla dui eu, aliquam velit. Fusce euismod lacus lectus, ac pretium nulla tempus sed. Morbi molestie varius ornare. Morbi ullamcorper faucibus erat. In imperdiet commodo sapien, ac bibendum nibh semper id. Mauris gravida tortor aliquam, ullamcorper purus ac, tincidunt magna. Nulla fermentum finibus augue, quis varius leo iaculis vitae. Ut porta sodales sem nec tincidunt. Suspendisse auctor vel ex ut elementum. Vivamus sit amet ornare massa, a condimentum nibh.","Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis arcu metus, tincidunt eu nisl a, aliquet vulputate libero. Pellentesque lobortis maximus tincidunt. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu ipsum tempor, fringilla dui eu, aliquam velit. Fusce euismod lacus lectus, ac pretium nulla tempus sed. Morbi molestie varius ornare. Morbi ullamcorper faucibus erat. In imperdiet commodo sapien, ac bibendum nibh semper id. Mauris gravida tortor aliquam, ullamcorper purus ac, tincidunt magna. Nulla fermentum finibus augue, quis varius leo iaculis vitae. Ut porta sodales sem nec tincidunt. Suspendisse auctor vel ex ut elementum. Vivamus sit amet ornare massa, a condimentum nibh."]]*3

    
    horizontal_gap = 0
    for cat_index in range(len(ex_list)):
        cat_title = df_features.columns[cat_index]
        vertical_gap = 0
        #draw cateogry title (i.e. df.column name)
        w, h = draw.textsize(cat_title)
        draw.text((WIDTH*cat_index+(WIDTH-w)/2,20), cat_title,  fill="black")
        
        
        for ex_index in reversed(range(len(ex_list[cat_index]))):
            
            #draw range type (i.e. top range, mid range, low range)
            
            rang_name = [ s + " range:" for s in ["low", "mid", "top"]][ex_index]
            w_r, h_r = draw.textsize(rang_name)
            draw.text((WIDTH*cat_index+(WIDTH-w_r)/2,EXAMPLE_OFFSET_Y+vertical_gap), rang_name,  fill="black")

            #draw each example below each other
            text = ex_list[cat_index][ex_index]
            lines = textwrap.wrap(text, width=MAX_CHARCTER_PER_LINE)
            y_text = h
            for line_index in range(min(len(lines),MAX_NR_LINES)):
                line = lines[line_index]
                width, height = draw.textsize(line)
                x = 170+((w - width) / 2)+horizontal_gap
                y = EXAMPLE_OFFSET_Y+y_text+vertical_gap
                draw.text((x, y), line, fill="black")
                y_text += height

            # After first block, add height of previous block and Y_GAP contstant as gap between blocks 
            vertical_gap+= h*MAX_NR_LINES+Y_GAP
        horizontal_gap += EXAMPLE_OFFSET_X+MAX_CHARCTER_PER_LINE*6
            
           
    
            
    img.save(CS.HOME_DIR+"text.png")


def df_to_plots(df):
    """Generate matplotlibs for each df column and save directly as report.png

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """
    lg.info("  drawing plot")
    
    df_features = df.drop("post_text",axis=1)
    nr_rows = len(list(df_features.columns))//5+1
    nr_cols =  len(list(df_features.columns))%100
    
    plt.rcParams["figure.figsize"] = (4*nr_cols,4*nr_rows)

    fig, axs = plt.subplots(nr_rows,nr_cols)
    for i in range(nr_cols-1):
            for j in range(nr_rows):
                index_sum = i+j
                data = df_features.iloc[:,index_sum].to_list()
                if axs.ndim > 1:
                    axs[i, j].hist(data)
                    axs[i, j].set_title(df_features.columns[index_sum])
                else:
                    axs[i].hist(data)
                    axs[i].set_title(df_features.columns[index_sum])
    

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(CS.HOME_DIR+"report.png")

    df_to_text_png(df)
