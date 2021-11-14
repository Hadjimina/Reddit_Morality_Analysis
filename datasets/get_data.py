import gdown
import os

def id_to_url(id):
    return 'https://drive.google.com/uc?id='+id

dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/data/"

files = {
    #"post_raw":{
    #    "url": id_to_url("17fvcKWGNHyxZFIU9kV-bZokhMSd4fEvt"),
    #    "output": dataset_dir+"posts_raw_27_6_2021.csv"
    #},
    "posts_cleaned":{
        "url":  id_to_url("1NBoO5CU4Dssya4IcWcjJ8Cx1rv-qiRuy"),
        "output": dataset_dir+"posts_cleaned_27_6_2021.csv"
    },
    "LIWC":{
        "url":  id_to_url("1lA8y_cto7vLYuBgTOkEn2lQueFVKHzSw"),
        "output": dataset_dir+"LIWC_27_6_2021.csv"
    },
    "moral_foundations":{
        "url":  id_to_url("1Vu188cUbUWA6A8wSTeDJPXfYGR2sGNmU"),
        "output": dataset_dir+"moral_foundations_27_6_2021.csv"
    },
    "comments_cleaned":{
        "url": id_to_url("15m_Kl7VScZgEaeyJyB3crdMB-9GeK-44"),
        "output": dataset_dir+"comments_clean_16_07_2021.csv"
    },
    #"comments_raw":{
    #    "url": id_to_url("12sJW2bnmOlXrmbIVPe00Zt2UQHSK7nd9"),
    #    "output": dataset_dir+"comments_raw_16_07_2021.csv"
    #},
        "moral_foundations_title":{
        "url":  id_to_url("121QNFJoLooNpsEFr58VauHsSQaMVlr7M"),
        "output": dataset_dir+"moral_foundations_title_27_6_2021.csv"
    }, 
     "LIWC_title":{
        "url":  id_to_url("12-D8MquT8jvuo489l718c5-hBaH82Qc9"),
        "output": dataset_dir+"LIWC_title_27_6_2021.csv"
    },

}

 
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
    
print("\nWhich would you like to download?\n")
keys = list(files.keys())
for i in range(len(keys)):
    split_output = files[keys[i]]["output"].split("/")
    out_name = split_output[len(split_output)-1]
    print("[{0}]: {1} ({2})".format(i, keys[i] ,out_name))


#print("[3]: Raw comments ("+comments["raw"]["output"]+")")
print("[ ]: ALL \n")

download_selection = input()

url, output = None, None

if download_selection:
    
    if download_selection.isnumeric() and int(download_selection) < len(keys):
        download_selection = int(download_selection)
        url = files[keys[download_selection]]["url" ]
        output = files[keys[download_selection]]["output" ]
    
    print("Downloading "+output+"\n")
    gdown.download(url, output, quiet=False)
else:
    print("Downloading ALL\n")
    for i in range(len(keys)):
        url = files[keys[i]]["url" ]
        output = files[keys[i]]["output" ]
        print("Downloading "+output+"\n")
        gdown.download(url, output, quiet=False)


