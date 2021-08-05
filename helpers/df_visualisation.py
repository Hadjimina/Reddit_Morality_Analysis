import coloredlogs
from pandas.core.frame import DataFrame
import constants as CS
import matplotlib.pyplot as plt
import logging as lg
coloredlogs.install()
from PIL import Image, ImageDraw, ImageFont
import textwrap

def df_to_text_png(df):
    """Generate png that shows 3 post example of each df column 

    returns : nothing, 

    Parameters
    ----------
    df : Dataframe
        dataframe with potentialy multiple columns    
    """
    WIDTH = 400
    HEIGHT = 400
    RANGE = 0.1
    SAMPLE_FRAC = 1

    lg.info("  drawing post examples")
    W, H = (WIDTH*(len(df.columns)-1),HEIGHT)
    img = Image.new("RGBA",(W,H),"white")
    draw = ImageDraw.Draw(img)

    df_sampeled =df.sample(frac=SAMPLE_FRAC)
    #print(df)
    print(df_sampeled)

    """ Define low, mid and top ranges to display posts for for each column
    s.t. we only need to iterate once over entire sampled DataFrame """


    range_list = [] #[[low_range, mid_range, top_range],...], each range is a list of length 2
    for i in range(len(df.columns)-1):
        title = df.columns[i]
        w, h = draw.textsize(title)
        #arial = ImageFont.truetype("arial.ttf", 9)
        #w,h = arial.getsize(title)
        draw.text((WIDTH*i+(WIDTH-w)/2,20), title,  fill="black")
       
        
        max = df[title].idxmax()
        min = df[title].idxmin()
        length = max * RANGE
        low_range = [int(min), int(min+length)]
        mid_range = [int(max/2-length/2), int(max/2+length/2)]
        top_range = [int(max-length), int(max)]
        
        
        range_list.append([low_range, mid_range, top_range])

    example_list = [[False, False, False]]*len(range_list)
    print(range_list)
    for i, row in enumerate(df_sampeled.itertuples(), 1):
        for j in range(1,len(df_sampeled.columns)-1):
           
            value = row[j]
            

            #check if value is in any ranges (low, mid, top)
            for cur_range_idx in range(len(range_list[j])):

                cur_range_lower = (range_list[j][cur_range_idx][0]) - (cur_range_idx != 0)
                cur_range_upper = (range_list[j][cur_range_idx][1]) + (cur_range_idx == 0)
                print(value)
                print(cur_range_lower, cur_range_upper)
                
                if value in range(cur_range_lower, cur_range_upper):
                    if not example_list[j][cur_range_idx]:
                        print("setting")
                        example_list[j][cur_range_idx] = row[len(row)-1]

        
    print(example_list)
        

        
    

    
    
    


    
        
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

    nr_rows = len(list(df.columns))//5+1
    nr_cols =  len(list(df.columns))%5
    
    plt.rcParams["figure.figsize"] = (4*nr_cols,4*nr_rows)

    fig, axs = plt.subplots(nr_rows,nr_cols)
    for i in range(nr_cols-1):
            for j in range(nr_rows):
                index_sum = i+j
                data = df.iloc[:,index_sum].to_list()
                if axs.ndim > 1:
                    axs[i, j].hist(data)
                    axs[i, j].set_title(df.columns[index_sum])
                else:
                    axs[i].hist(data)
                    axs[i].set_title(df.columns[index_sum])
    

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(CS.HOME_DIR+"report.png")

    df_to_text_png(df)
