'''
-locate the data folder
-list all filenames

-create an empty dictionary
-find the full paths by merging data folder path  and filenames for all filenames and add this list to dictionary key "image_paths"
-find the all labels from filenames by deleting .png part. And add this list to dictionary key "labels"
-convert dictionary to dataframe using pandas
-convert df to csv file using .to_csv


'''
import os
import pandas as pd
DATA_FOLDER = './data/'
all_file_names_in_data_folder = sorted(os.listdir(DATA_FOLDER))
print(all_file_names_in_data_folder)


a= [DATA_FOLDER + str(x) for x in all_file_names_in_data_folder]
print(a)

b=[str(x)[:-4] for x in all_file_names_in_data_folder]
print(b)

captcha = { "image_file_paths": a ,
            "labels": b

}

df=pd.DataFrame(captcha, columns=["image_file_paths", "labels"])
df.to_csv("data_csv")


# './data/226md.png'
# 226md




