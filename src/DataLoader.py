from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
import pandas as pd

class Sample:
    "sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

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

def return_text(filename,k,train):
    
    while(train[k][1]!=int(filename[15])):
        k+=1
    k=k+int(filename[17])-1
    return train[k][3]


class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1]=='/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        
        
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        fimages = '../data/data/'
        _,filenames = load_images_from_folder(fimages)
        #print(filenames[0][5]+filenames[0][6]+filenames[0][7]+filenames[0][8]+filenames[0][9])
        #para=15 line=17 file=5-9
        xlfile = '../data/xldata.xlsx'
        xl = pd.ExcelFile(xlfile)
        #print(xl.sheet_names)
        df1 = xl.parse('train')
        train = df1.values.tolist()
        train[6421][0]='A0642'
        for t in train:
            t[0]=str(t[0]).strip()
        #print(train[0][3].encode("utf-8"))
        for filename in filenames:
            k=index_2d(train,filename[5]+filename[6]+filename[7]+filename[8]+filename[9])[0]
            atext=return_text(filename,k,train)
            fileName=filePath + 'data/'+filename
            if not os.path.getsize(fileName):
                bad_samples.append(filename)
                continue
            chars = chars.union(set(list(atext)))
            if(len(atext)<99):
                self.samples.append(Sample(atext, fileName))
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)
        
        
#        f=open(filePath+'words.txt', mode='r', encoding='utf-8')
#        chars = set()
#        bad_samples = []
#        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
#        for line in f:
#            # ignore comment line
#            if not line or line[0]=='#':
#                continue
#            #print(line.encode("utf-8"))
#            lineSplit = line.strip().split(' ')
#            #assert len(lineSplit) >= 9
#            
#            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
#            fileNameSplit = lineSplit[1]
#            fileName = filePath + 'data/'+fileNameSplit+'.tif'
#
#            # GT text are columns starting at 9
#            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
#            chars = chars.union(set(list(gtText)))
#
#            # check if image is not empty
#            if not os.path.getsize(fileName):
#                bad_samples.append(lineSplit[0] + '.png')
#                continue
#
#            # put sample into list
#            self.samples.append(Sample(gtText, fileName))
#
#        # some images in the IAM dataset are known to be damaged, don't show warning for them
#        if set(bad_samples) != set(bad_samples_reference):
#            print("Warning, damaged images found:", bad_samples)
#            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training 
        self.numTrainSamplesPerEpoch = 25000 
        
        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))


    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    
    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)
        
        
    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
        #j=0
        #for i in imgs:
        #    cv2.imwrite( str(j)+'_pre3.png', i )
        #    j = j+1
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)


