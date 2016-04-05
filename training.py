#!/usr/bin/env python
import os, sys
from skimage import data, io, filters,feature,color
from skimage.viewer import ImageViewer
import numpy as np
import pickle
from sklearn.svm import LinearSVC


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

def dumpdata():
    with open("training_label.txt", 'wb') as f:
        pickle.dump(label_list, f)
    with open("training_data.txt","wb") as f:
        pickle.dump(data_list, f)



    """f=open("training_label","w")
    for i in label_list:
        f.write(str(i)+"\n")
    f.close()
    f=open("training_data","w")
    for i in data_list:
        f.write(str(i)+"\n")
    f.close()"""


# the data and label lists
label_list = []
data_list = []
# initialize the local binary patterns descriptor along with
lbpDesc = LocalBinaryPatterns(24, 8)

def collectdata():
    global label_list
    global data_list
    basepath_training="/home/partha/projects/ageidentification/images/training"
    count=0
    for fname_training in os.listdir(basepath_training):
        count+=1
        path_class=os.path.join(basepath_training, fname_training)
        """This path_class is our age group we are training our svm for"""
        """complete training code will be inside this part"""
        print "Printing the contents of folder"+path_class+"\n"
        class_name= path_class.split("/")[-1]
        #now identify which class this folder belong to
        if class_name=="1-10":
            class_index = 0
        elif class_name=="11-20":
            class_index = 1
        elif class_name == "21-30":
            class_index = 2
        elif class_name == "31-40":
            class_index = 3
        elif class_name == "41-50":
            class_index = 4
        elif class_name == "51-60":
            class_index = 5
        elif class_name == "61-70":
            class_index = 6
        print "Am training class"+ str(class_index)+"\n"
        for file in os.listdir(path_class):
            """Now the actual processing of the file and training our svm like wise"""
            #print file+"\n"
            base_save="/home/partha/projects/ageidentification/dump/"
            img = io.imread(path_class+"/"+file)
            grayimg = color.rgb2gray(img)
            #io.imsave(base_save+file,grayimg)
            count+=1
            histogramdata = lbpDesc.calculatehistogram(grayimg)
            #add the labels and data
            label_list.append(class_index)
            data_list.append(histogramdata)
    #now need to dump this data so that we can train our system anytime
    print "The number of image processed="+str(count)+"\n"
    dumpdata()


#program execution
collectdata()


