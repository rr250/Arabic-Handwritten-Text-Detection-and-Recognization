# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:52:04 2019

@author: Rishabh.ranjan
"""
import sys
import logging
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import pandas as pd


#xlfile = '../data/xldata.xlsx'
#words = '../data/validate.txt'
#xl = pd.ExcelFile(xlfile)
#print(xl.sheet_names)
#df1 = xl.parse('validate')
#train = df1.values.tolist()
#print(train[0][3])
#f=open(words, "a+", encoding="utf-8")
#for i in range(3207):
#    f.write('a'+str(i)+' '+'ok 176 957 1104 219 51 NN ')
#    f.write(train[i][3])
#f.close()
#
#
#k=index_2d(train,filenames[21][5]+filenames[21][6]+filenames[21][7]+filenames[21][8]+filenames[21][9])[0]
#while(train[k][1]!=int(filenames[21][15])):
#    k+=1
#k=k+int(filenames[21][17])-1
#print(train[k])
#print(k)

def index_2d(data, search):
    for i, e in enumerate(data):
        try:
            return i, e.index(search)
        except ValueError:
            pass
    raise ValueError("{} is not in list".format(repr(search)))

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images,filenames

def return_text(filename,k):
    
    while(train[k][1]!=int(filename[15])):
        k+=1
    k=k+int(filename[17])-1
    return train[k][3]
    
    
    
fimages = '../data/data/'
_,filenames = load_images_from_folder(fimages)
print(filenames[0][5]+filenames[0][6]+filenames[0][7]+filenames[0][8]+filenames[0][9])
#para=15 line=17 file=5-9
xlfile = '../data/xldata.xlsx'
words = '../data/words1.txt'
f=open(words, "a+", encoding="utf-8")
xl = pd.ExcelFile(xlfile)
print(xl.sheet_names)
df1 = xl.parse('train')
train = df1.values.tolist()
train[6421][0]='A0642'
for t in train:
    t[0]=str(t[0]).strip()
print(train[0][3])
f=open(words, "a+", encoding="utf-8")
for filename in filenames:
    k=index_2d(train,filename[5]+filename[6]+filename[7]+filename[8]+filename[9])[0]
    atext=return_text(filename,k)
    f.write(atext)
f.close()























