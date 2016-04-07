import sys

import dlib
from skimage import io
detector = dlib.get_frontal_face_detector()
img = io.imread('s1.jpg')
faces = detector(img)
for d in faces:
    print "left,top,right,bottom:", d.left(), d.top(), d.right(), d.bottom()
    cropped=img[d.top()-200:d.bottom()+200,d.left()-100:d.right()+100]
    io.imsave("set.jpg", cropped)
