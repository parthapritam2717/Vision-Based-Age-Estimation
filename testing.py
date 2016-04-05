#!/usr/bin/env python
import os, sys
from skimage import data, io, filters,feature,color
from skimage.viewer import ImageViewer
import numpy as np
from sklearn.svm import LinearSVC
import pickle

class LocalBinaryPatterns:
    """This class is to define the LBP value of a pixel"""
    def __init__(self, numberOfPoints, radius):
        self.numberOfPoints = numberOfPoints
        self.radius = radius

    def calculatehistogram(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numberOfPoints, self.radius, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(),
                                      bins=np.arange(0, self.numberOfPoints + 3),
                                      range=(0, self.numberOfPoints + 2))
        #now we need to normalise the histogram so that the total sum=1
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)
        return histogram




age_list={0:"0-10",1:"11-20",2:"21-30",3:"31-40",4:"41-50",5:"51-60",6:"61-70"}

# the data and label lists
label_list = []
data_list = []
model = LinearSVC(C=100.0, random_state=42)

def trainSystem():
    global  model
    global data_list
    global label_list
    #read the training data and the training label from the text file
    """f=open("training_label","r")
    filecontents = f.readlines()
    for line in filecontents:
        foo = line.strip('\n')
        label_list.append(int (foo))
    f.close()
    f=open("training_data","r")
    filecontents=f.readlines()
    for line in filecontents:
        foo=line.strip("\n")
        data_list.append(float(foo))
    f.close()"""


    with open("training_label.txt", 'rb') as f:
        label_list = pickle.load(f)
    with open("training_data.txt", 'rb') as f:
        data_list = pickle.load(f)

    #now train the linear SVM model for classification of age
    model.fit(data_list, label_list)


def testsystem():
    global model
    lbpDesc=LocalBinaryPatterns(24, 8)
    basepath_testing = "/home/partha/projects/ageidentification/images/testing"
    #print label_list
    #print data_list
    for fname_training in os.listdir(basepath_testing):
        #print fname_training
        #do the image processing calculate the LBp for the test image and then find which class it belong
        img = io.imread(basepath_testing + "/" + fname_training)
        grayimg = color.rgb2gray(img)
        histogramdata = lbpDesc.calculatehistogram(grayimg)
        prediction = model.predict(histogramdata)[0]
        print "The image "+str(fname_training)+" belongs to the class "+ str(prediction)+" and age is approx "+str(age_list[prediction])

#program execution
trainSystem()
testsystem()
