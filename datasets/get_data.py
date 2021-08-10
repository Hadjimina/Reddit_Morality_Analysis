import gdown
import os

def id_to_url(id):
    return 'https://drive.google.com/uc?id='+id

dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/data/"
print(dataset_dir)

posts = {
    "raw":{
        "url": id_to_url("17fvcKWGNHyxZFIU9kV-bZokhMSd4fEvt"),
        "output": dataset_dir+"posts_27_6_2021.csv"
    },
    "cleaned":{
        "url":  id_to_url("1NBoO5CU4Dssya4IcWcjJ8Cx1rv-qiRuy"),
        "output": dataset_dir+"posts_cleaned_27_6_2021.csv"
    }
}

comments={
    "raw":{
        "url": id_to_url("18qxC5bv2oGotWPVUYvvSHUWtrIob-QUA"),
        "output": dataset_dir+"comments_16_07_2021.csv"
    }
}
 
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
    
print("Which would you like to download?\n")
print("[1]: Raw posts ("+posts["raw"]["output"]+")")
print("[2]: Clean posts ("+posts["cleaned"]["output"]+")")
print("[3]: Raw comments ("+comments["raw"]["output"]+")")
print("[ ]: ALL \n")

download_selection = input()

url, output = None, None

if download_selection == "1":
    url = posts["raw" ]["url" ]
    output = posts["raw" ]["output" ]
elif download_selection == "2":
    url = posts["cleaned" ]["url" ]
    output = posts["cleaned" ]["output" ]
elif download_selection == "3":
    url = comments["raw" ]["url" ]
    output = comments["raw" ]["output" ]
else:
    print("\nStart download of "+posts["raw" ]["output" ]+"\n")
    gdown.download(posts["raw" ]["url" ], posts["raw" ]["output" ], quiet=False)

    print("\nStart download of "+posts["cleaned" ]["output" ]+"\n")
    gdown.download(posts["cleaned" ]["url" ], posts["cleaned" ]["output" ], quiet=False)

    print("\nStart download of "+comments["raw" ]["output" ]+"\n")
    gdown.download(comments["raw" ]["url" ], comments["raw" ]["output" ], quiet=False)    

if not output == None:
    print("Downloading "+output+"\n")
    gdown.download(url, output, quiet=False)