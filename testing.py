#!/usr/bin/env python
import os
import pickle
import warnings

import dlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, feature, color
from skimage import novice
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


def fxn():
    warnings.warn("deprecated", DeprecationWarning)
separator=os.sep

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




def testsystem():
    detector = dlib.get_frontal_face_detector()

    global model
    lbpDesc=LocalBinaryPatterns(24, 8)
    basepath_testing = os.getcwd()  # "/home/partha/projects/ageidentification/images/testing"
    bb = "images"
    basepath_temp = os.path.join(basepath_testing, bb)
    basepath_temp = os.path.join(basepath_temp, "testing")
    basepath_testing = basepath_temp
    base_save = os.path.join(os.getcwd(), "dumptest")
    for fname_training in os.listdir(basepath_testing):
        #do the image processing calculate the LBp for the test image and then find which class it belong
        img = io.imread(basepath_testing + separator + fname_training)
        img_temp = novice.open(basepath_testing + separator + fname_training)
        width,height=img_temp.size
        print width,height
        """face detection and cropping"""
        faces = detector(img)
        count=0
        for d in faces:
            d_top = d.top()
            d_bottom = d.bottom()
            d_left = d.left()
            d_right = d.right()
            print "left,top,right,bottom:", d.left(), d.top(), d.right(), d.bottom()

            if d.top() > 100:
                d_top = d.top() - 100
            else:
                d_top = 0
            if height - d.bottom() > 100:
                d_bottom = d.bottom() + 100
            else:
                d_bottom = height
            if d.left() > 100:
                d_left = d.left() - 100
            else:
                d_left = 0
            if width - d.right() > 100:
                d_right = d.right() + 100
            else:
                d_right = width
            print d_top,d_bottom,d_left,d_right
            count+=1
            #cropped = img[d.top():d.bottom(), d.left():d.right()]
            cropped = img[d_top:d_bottom, d_left:d_right]
            if(width>800 and height>900):
                #io.imsave(base_save +str(count)+ fname_training, cropped)
                grayimg = color.rgb2gray(cropped)
                histogramdata = lbpDesc.calculatehistogram(grayimg)
                warnings.filterwarnings("ignore")
                """This line is to supress the warning"""
                prediction = model.predict(histogramdata)[0]
                print "The image "+str(fname_training)+" age is approx "+str(age_list[prediction])
                plt.imshow(cropped)
                plt.show()
            else:
                #io.imsave(base_save + str(count)+fname_training, img)
                grayimg = color.rgb2gray(img)
                histogramdata = lbpDesc.calculatehistogram(grayimg)
                warnings.filterwarnings("ignore")
                """This line is to supress the warning"""
                prediction = model.predict(histogramdata)[0]
                print "The image " + str(fname_training) + " age is approx " + str(age_list[prediction])
                plt.imshow(img)
                plt.show()
#program execution
model=joblib.load("agemodel.txt")
testsystem()
