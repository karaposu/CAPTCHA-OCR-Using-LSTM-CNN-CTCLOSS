# ----------------------------- DATA & LBL PREP
import pandas as pd
import os
from pathlib import Path
from sklearn import preprocessing
from sklearn import model_selection
import numpy as np

'''
- read the csv file using pandas
- use iloc to read file paths
- convert image paths strings to Path type objects
- use iloc to get targets as list
- split each target to characters
- merge all characters of all targets and feed it to label encoder
- encode targets using label encoder

'''


def Extract_Data():
    annotations = pd.read_csv("data_csv")
    image_files_paths = []


    image_files_paths= [Path(x)  for x in annotations.iloc[:, 1]]



    targets_orig = annotations.iloc[:, 2]

    labels = targets_orig.tolist()

    targets = [[c for c in x] for x in labels]

    targets_flat = [c for clist in targets for c in clist]

    # making encoder eat all letters to find the alphabet.

    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(targets_flat)



    # converting  all lbls to numbers according to alphabet
    targets_enc = [LabelEncoder.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)



    return (image_files_paths ,targets_enc , LabelEncoder)







#
#
# import pandas as pd
# from pathlib import Path
# from sklearn import preprocessing
# import numpy as np
#
# def Extract_Data():
#     annotations = pd.read_csv("data_csv")
#     image_file_paths=[]
#
#     image_file_paths = [Path(x) for x in annotations.iloc[:,1]]
#     print(image_file_paths)
#     print(len(image_file_paths))
#
#
#     targets_raw= annotations.iloc[:,2].tolist()
#     print(targets_raw)
#
#     targets= [[e for e in x]for x in targets_raw]
#     print(targets)
#
#     all_characters=[]
#     for x in targets:
#         for e in x:
#             all_characters.append(e)
#
#     LabelEncoder= preprocessing.LabelEncoder()
#     LabelEncoder.fit(all_characters)
#
#     print(all_characters)
#     print(LabelEncoder.classes_)
#
#     targets_encoded= [LabelEncoder.transform(x) for x in targets]
#     targets_encoded= np.array(targets_encoded)
#
#     print(targets_encoded)
#
#
#     return (image_file_paths, targets_encoded, LabelEncoder)
#
#
#
#
# if __name__ == "__main__":
#    Extract_Data()

#
#
#
#
#
#
#
#
#
