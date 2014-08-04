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

MAX_VIDEO_SIZE = 1000 #Frames

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
        
    img = img[roi[0]:roi[1],roi[2]:roi[3]]  # vert:vert, hor:hor
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # image, threshold1, threshold2, apertureSize
    edges = cv2.Canny(gray,65,180,apertureSize = 3)

    # Save debugging images
    if debug:
        roiPath = outPath + "\\" + ident + "-1ROI.png"
        cannyPath = outPath + "\\" + ident + "-2Canny.png"
        overlayPath = outPath + "\\" + ident + "-3Overlay.png"
        cv2.imwrite(roiPath,img)
        cv2.imwrite(cannyPath,edges)
    
    lines = cv2.HoughLines(edges,2,np.pi/1080,30)
    
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


def videoToFilteredVideo(videoPath, outPath, N, deg, roi):
    """Filters and saves every Nth image in a video with getContactAngle()
    
    Input:
    videoPath = a string, path of video to be analyzed
    outPath = a string, working directory
    N = a positive integer, images divisable by it will be analyzed
    
    Output:
    A cropped, filtered, rotated image
    """
    
    if N <= 0:
        print "N must be an integer >0."
        return None

    count = 0
    
    # Get rotation matrix centered in roi
    x_cent = int((roi[0]+roi[1])/2)
    y_cent = int((roi[2]+roi[3])/2)
    rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),deg,1)
    print x_cent, y_cent, deg
    print rot_Matrx
    
    # Open output video    
    height = roi[1]-roi[0]
    width = roi[3]-roi[2]
    outName = outPath + 'Filtered.avi'
    
    # Output Filename, compression, fps, (frame_W, frame_H)
    
    # See notes below for possible compression formats
    video = cv2.VideoWriter(outName, cv2.cv.CV_FOURCC('I','4','2','0')
                            ,25,(width,height))

    # Start reading input video
    cap = cv2.VideoCapture(videoPath)
        
    while True:
        ret, frame = cap.read()
        
        # If frameNum/N=0 is returned do analysis
        if ret and (count % N) == 0:          
                        
            imgNum = string.zfill(count,4)
            
            rows,cols,channels = frame.shape            
            print ret, imgNum, cols, rows, channels
            
            # Rotate and crop image based on Deg / Roi
            img = cv2.warpAffine(frame,rot_Matrx,(cols,rows))            
            img_Nom = img[roi[0]:roi[1],roi[2]:roi[3]]  # vert:vert, hor:hor         
            
            # Gray scale and apply a mask to remove unecessary points
            img = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) # convert to GS
            rows, cols = img.shape
            mask = np.zeros([rows,cols], np.uint8)
            mask[:,:90] = 255
            mask[:,280:] = 255
            img = cv2.add(img,mask)

            # Adjust contrast... this is difficult to get right
            img = cv2.add(img,90)
            r,img = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
            
            # Convert Nom to grayscale so it can output in the same png
            img_Nom = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) # convert to GS 
            img_Debug = np.hstack((img,img_Nom)) # Side by side for contrast          

            cv2.imwrite(outPath+imgNum+'.png', img_Debug)

            # Convert back to BGR for video codec
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) # convert to GS  
            video.write(img)
            
        if count > MAX_VIDEO_SIZE or ret == False:
            print count
            break
        
        count += 1

    cv2.destroyAllWindows()
    video.release()
    

def osWalk(outPath, roi):
    """ Use os.walk to do stuff
    """
    height = roi[1]-roi[0]
    width = roi[3]-roi[2]
    outName = outPath + '\\CompileVideo.avi'
    
    # Possible compression formats: M J P G, P I M 1, I 4 2 0
    # Translation: Mjpg, i dont know, uncompressed avi
    video = cv2.VideoWriter(outName, cv2.cv.CV_FOURCC('P','I','M','1'),
                            25,(width,height))

    for (root, subFolders, files) in os.walk(outPath):
    
        #Second condition forces this to run in only the index dir
        for i, item in enumerate(files):
            print item, root            
            if '.png' in item:
                img = cv2.imread(root+'\\'+item)                
                video.write(img)

    #cv2.destroyAllWindows()
    video.release()

# Filter video and resave it
inputVid="D:\\GitHub\\A2_work\\20140728-14_30_42-CA1_M01SS-27.MOV"
outputPath="D:\\GitHub\\A2_work\\Work\\"
rotation = 2 # degrees
roi = (850,1080,800,1175)  # xmin:xmax (vert, row), ymin:ymax (horz, col)
N = 10
print inputVid
print outputPath
videoToFilteredVideo(inputVid, outputPath, N, rotation, roi)


"""
NOTES:

roi has a minimum size: opencv video.write seems to have a minimum image size

FOURCC CODECs
int fourCC_code = CV_FOURCC('M','J','P','G');	// M-JPEG codec (may not 
be reliable). 
int fourCC_code = CV_FOURCC('P','I','M','1');	// MPEG 1 codec. 
int fourCC_code = CV_FOURCC('D','I','V','3');	// MPEG 4.3 codec. 
int fourCC_code = CV_FOURCC('I','2','6','3');	// H263I codec. 
int fourCC_code = CV_FOURCC('F','L','V','1');	// FLV1 codec. 

MASK LOGIC (CMPOP)
cv2.compare(src1, src2, cmpop[, dst])
CMP_EQ src1 is equal to src2.
CMP_GT src1 is greater than src2.
CMP_GE src1 is greater than or equal to src2.
CMP_LT src1 is less than src2.
CMP_LE src1 is less than or equal to src2.
CMP_NE src1 is unequal to src2.
"""
