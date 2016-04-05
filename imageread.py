#!/usr/bin/env python
from skimage import data, io, filters,feature,color
from skimage.viewer import ImageViewer
import os
img = io.imread("s1.jpg")
new = color.rgb2gray(img)
io.imsave("s2.jpg",new)






