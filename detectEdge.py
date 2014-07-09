# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 21:55:27 2014

@author: Taylor Cooper

Demonstrating basic image subtraction
"""

import numpy as np
import cv2
import os
import string
from pylab import *

MAX_VIDEO_SIZE = 500 #Frames

def getContactAngle(filePath, ident):
    """Estimates contact angle using hough lines.
    """
    roiPath = pathHough + "/" + ident + "-1ROI.png"
    cannyPath = pathHough + "/" + ident + "-2Canny.png"
    overlayPath = pathHough + "/" + ident + "-3Overlay.png"
    
    img = cv2.imread(filePath)
    img = img[450:600, 700:800] # Set ROI
    cv2.imwrite(roiPath,img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,30,70,apertureSize = 3)
    cv2.imwrite(cannyPath,edges)
    
    lines = cv2.HoughLines(edges,2,np.pi/1080,20)
    
    if lines == None:
        print "houghLines dropout:", ident
        return None
    
    threshLines = []
    contactAngleLines = []
    
    for i, item in enumerate(lines[0]):
        if 0 < item[1] < 0.95*np.pi/2:
            contactAngleLines.append(item[1])
            threshLines.append(item)

    if threshLines == []:
        print "threshLines dropout:", ident
        return None            
    
    for rho,theta in threshLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),1)


    ca = np.pi/2 - np.mean(contactAngleLines)
    caDeg = ca*180/np.pi
    
    linesPath = pathHough + "/" + ident + "-4Lines_"+ str(int(caDeg)) +".png"

    return caDeg
    
    cv2.imwrite(overlayPath, edges)
    cv2.imwrite(linesPath,img)
    



def videoToImages():
    """May work need to get a copy of that video to see for myself.    
    
    Alternatives to cv2.VideoCaptrure:
    
    1) Extracting images from a video:
    ffmpeg -i foo.avi -r 1 -s WxH -f image2 foo-%03d.jpeg
    
    2) Creating images from a video
    ffmpeg -f image2 -i foo-%03d.jpeg -r 12 -s WxH foo.avi
    
    Then call these with os.system()
    """
    cap = cv2.VideoCapture(path+"/drop/drop.avi")
    
    print cap.isOpened()
    count = 0
    
    while True:
        ret, frame = cap.read()
        print ret    
        
        if ret and (count % 10) == 0:
            name = "contactAnglePicture-" + string.zfill(count,3) + ".png"
            cv2.imwrite(name, frame)        
        
        if count > MAX_VIDEO_SIZE:
            print count
            break
        
        count += 1

def measureMultipleImages():
    """ Use os.walk to measure all images in the folder
    """

    contactAngles = []
    
    for (root, subFolders, files) in os.walk(pathCA):
    
        #Second condition forces this to run in only the index dir
        for i, item in enumerate(files):
            if '.png' in item:
                print item
                angle = getContactAngle(pathCA+"/"+item, item[8:11])
                if angle:
                    contactAngles.append(angle)
                
    plot(arange(len(contactAngles)), contactAngles)
    savefig("contactAngles.png")
    
            
path = os.getcwd()
pathCA = path + "/contact_angle"
pathHough = path + "/houghoutputs"

print pathCA
print pathHough
measureMultipleImages()

"""
Potentially useful code for making a video from images,  video it generates
are not viable unfortunately.

img1 = cv2.imread(pathCA+"\\goodVid-001.png")
#img2 = cv2.imread(pathCA+"\\goodVid-002.png")
#img3 = cv2.imread(pathCA+"\\goodVid-003.png")
#video.write(img1)
#video.write(img2)
#video.write(img3)

height, width, layers = img1.shape

print height, width

#video = cv2.VideoWriter(pathCA+"\\goodVid.avi",-1,1,(width,height))
#
#for root, subFolders, files in os.walk(pathCA):
#    
#    #Second condition forces this to run in only the index dir
#    for i, item in enumerate(files):
#        if '.png' in item:
#            img = cv2.imread(pathCA+"\\"+item)
#            print item
#            video.write(img)
#        
#cv2.destroyAllWindows()
#video.release()
"""
