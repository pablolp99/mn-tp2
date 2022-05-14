import pandas as pd
import sys

sys.path.insert(1, '/home/pablo/UBA/comp2022/MN/mn-tp2/build')

print("Importing mnpkg")

from mnpkg import *

print("Reading CSV")

df = pd.read_csv("../data/train.csv")
labels = df["label"]

img_df = df.drop(columns='label')
imgs = []
for i in range(len(img_df)):
    imgs.append(img_df.iloc[i].tolist())

print("Reading IMGS")

matrix = read_img(imgs, len(imgs), len(imgs[0]))

print("Success")