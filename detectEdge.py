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

MAX_VIDEO_SIZE = 100 #Frames

def getContactAngle(inputImg, outPath, ident, roi, debug=True):
    """Estimates contact angle using hough lines.

    Inputs:
    inputImg = string/file location or opencv img variable (np.array)
    outPath = working directory
    ident = image number
    roi = region of interest
    
    Output:
    pl
    """

    # If input file type is a string read that file location, else image   
    if isinstance(inputImg, basestring):
        img = cv2.imread(inputImg)
    else:
        img = inputImg
        
    img = img[roi[0]:roi[1],roi[2]:roi[3]]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,30,70,apertureSize = 3)

    # Save debugging images
    if debug:
        roiPath = outPath + "\\" + ident + "-1ROI.png"
        cannyPath = outPath + "\\" + ident + "-2Canny.png"
        overlayPath = outPath + "\\" + ident + "-3Overlay.png"
        cv2.imwrite(roiPath,img)
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

    # Calculate Contact Angle
    ca = np.pi/2 - np.mean(contactAngleLines)
    caDeg = ca*180/np.pi
    
    # Save debugging images
    if debug:
        for rho,theta in threshLines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        
            cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),1)
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    
        linesPath = outPath + "\\" +ident+ "-4Lines_"+str(int(caDeg))+".png"
        cv2.imwrite(linesPath,img)
        cv2.imwrite(overlayPath, edges)
     
    return caDeg # Return contact angle in degrees


def videoToImages(videoPath, outPath):
    """Works will split every 10th image out and save it to the outPath.

    Ideally this would be done in a for loop for some reason it doesn't work.
    """
    cap = cv2.VideoCapture(videoPath)
    
    print cap.isOpened()
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        print ret
        
        if ret and (count % 10) == 0:
            imgNum = string.zfill(count,3)
            name = outputPath + "contactAnglePicture-" + imgNum + ".png"
            cv2.imwrite(name, frame)        
        
        if count > MAX_VIDEO_SIZE or ret == False:
            print count
            break
        
        count += 1

def measureMultipleImages(inputPath, outputPath):
    """ Use os.walk to measure all images in the folder
    """

    contactAngles = []
    
    for (root, subFolders, files) in os.walk(inputPath):
    
        #Second condition forces this to run in only the index dir
        for i, item in enumerate(files):
            if '.png' in item:
                print item
                angle = getContactAngle(inputPath+"\\"+item, outputPath,
                                        item[8:11])
                if angle:
                    contactAngles.append(angle)
                
    plot(arange(len(contactAngles)), contactAngles)
    savefig("contactAngles.png")


    
def videoToContactAngles(videoPath, outPath, N, roi):
    """Analyzes every Nth image in a video with getContactAngle()
    
    Input:
    videoPath = a string, path of video to be analyzed
    outPath = a string, working directory
    N = a positive integer, images divisable by it will be analyzed
    
    Output:
    plot of CA vs image number
    if debug is enabled the houghlines plots will be output in the working dir
    """
    cap = cv2.VideoCapture(videoPath)
    
    if N <= 0:
        return "N must be an integer >0."
    
    count = 0
    contactAngles = []
    frameNumbers = []    
    
    while True:
        ret, frame = cap.read()
        
        # If frameNum/N=0 is returned do analysis
        if ret and (count % N) == 0:          
                        
            imgNum = string.zfill(count,4)
            print ret, imgNum
            
            angle = getContactAngle(frame, outPath, imgNum, roi)
            if angle:
                    contactAngles.append(angle)
                    frameNumbers.append(count)
                    
        if count > MAX_VIDEO_SIZE or ret == False:
            print count
            break
        
        count += 1

    plot(frameNumbers, contactAngles)
    savefig(outPath+"\\contactAngles.png")
            
# Basic functionality            
#path = "C:\\Users\Taylor\\Documents\\GitHub\\MW_ContactAngleMeasurement"
#pathCA = path + "\\contact_angle"
#pathHough = path + "\\houghoutputs"
#print pathCA
#print pathHough
#roi = (450,600,700,800)
#measureMultipleImages(pathCA, pathHough, roi)
            
# Basic Spliting video to images
#inputVid = "C:\\Users\\Taylor\\Documents\\GitHub\\MW_ContactAngleMeasurement\\\
#drop\\drop.avi"
#outputPath = "C:\\Users\\Taylor\\Documents\\GitHub\\MW_ContactAngleMeasurement\
#\\drop\\"
#print inputVid
#print outputPath
#videoToImages(inputVid, outputPath)

# Combined, split video and analyze it
inputVid = "D:\\Dropbox\\Clients\\AFCC\\Projects\\Metal FSU\\\
A2 - Material Characterization\\TEST DATA\TEST CA1\\20140725_GizmoDry_Analysis\
\\2014_07_28\\20140728-14_30_42-CA1_M01SS-27.MOV"
outputPath = "D:\\Dropbox\\Clients\\AFCC\\Projects\\Metal FSU\\\
A2 - Material Characterization\\TEST DATA\TEST CA1\\20140725_GizmoDry_Analysis\
\\2014_07_28\\work"
roi = (200,1800,400,1050)
N = 10
print inputVid
print outputPath
videoToContactAngles(inputVid, outputPath, N, roi)



"""
NOTES:

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

    
    Alternatives to cv2.VideoCaptrure:
    
    1) Extracting images from a video:
    ffmpeg -i foo.avi -r 1 -s WxH -f image2 foo-%03d.jpeg
    
    2) Creating images from a video
    ffmpeg -f image2 -i foo-%03d.jpeg -r 12 -s WxH foo.avi
    
    Then call these with os.system()
"""
