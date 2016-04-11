#!/usr/bin/env python
import os
import pickle

import dlib
import numpy as np
from skimage import io, feature, color
from skimage import novice
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

detector = dlib.get_frontal_face_detector()

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

def dumpdata():
    with open("training_label.txt", 'wb') as f:
        pickle.dump(label_list, f)
    with open("training_data.txt","wb") as f:
        pickle.dump(data_list, f)


# the data and label lists
label_list = []
data_list = []
# initialize the local binary patterns descriptor along with
lbpDesc = LocalBinaryPatterns(24, 8)
count = 0
model = LinearSVC(C=100.0, random_state=42)
def collectdata():
    global label_list
    global data_list
    basepath_training=os.getcwd()#"/home/partha/projects/ageidentification/images/training_age"
    bb = "images"
    basepath_temp = os.path.join(basepath_training, bb)
    bbc = "training_age"
    basepath_temp = os.path.join(basepath_temp, bbc)
    # print basepath_temp
    basepath_training = basepath_temp
    global count
    for fname_training in os.listdir(basepath_training):

        path_class=os.path.join(basepath_training, fname_training)
        """This path_class is our age group we are training our svm for"""
        """complete training code will be inside this part"""
        print "Printing the contents of folder"+path_class+"\n"
        class_name= path_class.split(separator)[-1]
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
            base_save=os.getcwd()
            base_save=os.path.join(base_save,"dump")
            img = io.imread(path_class+separator+file)
            img_temp = novice.open(path_class+separator+file)
            width, height = img_temp.size
            #print width, height

            faces = detector(img)

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
                count+=1
                #print d_top, d_bottom, d_left, d_right
                #cropped = img[d.top():d.bottom(), d.left():d.right()]
                cropped = img[d_top:d_bottom, d_left:d_right]
                if (width > 800 and height > 900):
                    grayimg = color.rgb2gray(cropped)
                else:
                    grayimg = color.rgb2gray(img)
                #io.imsave(base_save+str(count)+file,grayimg)

                histogramdata = lbpDesc.calculatehistogram(grayimg)
                #add the labels and data
                label_list.append(class_index)
                data_list.append(histogramdata)
    #now need to dump this data so that we can train our system anytime
    model.fit(data_list, label_list)
    joblib.dump(model,"agemodel.txt")
    print "The number of image processed="+str(count)+"\n"
    #dumpdata()
#program execution
collectdata()


