# -*- coding: utf-8 -*-

"""
    Author:
        Taylor Cooper
    Description:
        Estimates contact angle from goniometer videos.              
    Date Created:
        Aug 4, 2014
    
    Arguments and Inputs:
        config.py:
        PATH - videoPath/filePath - a single .MOV or file structure with .MOVs
        VSF - video start frame - Starting frame for analysis
        VEF - video end frame - Ending frame for analysis
        N - frames analyzed - Ratio of frames analyzed to frames ignored
        R - rotation - Rotate video by this angle before analyzing
        ROI - roi - Region of interest
        MASK - mask - Region masked (basically max x extent of bubble)
        BR - brightness - Value added to brighten images
        TH - threshold - Value used to binary threshold images
        BED - bubble edge depth - (BED) Interface between bubble and surface
                        
    Outputs:
        .avi of cropped and rotated original video
        .avi of filtered original video with BED, quadratics, contact angles (CA)
        plot of CA vs image number
        .txt of data used in CA vs image number
    Dependencies:
        numpy, cv2, os, string, pylab
                  
    History:                  
    --------------------------------------------------------------
    Date:    
    Author:    Taylor Cooper
    Modification:    
     --------------------------------------------------------------
"""

import numpy as np
import cv2, os, string
from pylab import *
import config

MAX_VIDEO_SIZE = 1000 #Frames

default = dict(
    PATH = 'D:\\GitHub\\workspace\\A2_work\\20140728-14_30_42-CA1_M01SS-27.MOV',
    VSF = 80,
    VEF = 660,
    N = 10,
    R = 2,
    ROI = (850,1080,800,1175), # row_min, row_max, col_min, col_max
    MASK = (90,280,255), # col_min, col_max, value outside mask
    BR = 90,
    TH = 160,
    BED = 150, # row (0 is at top of image)
)

def findEdge(arr):
    # j is position in arr, pxVal is value of the pixel
    for j,pxVal in enumerate(np.nditer(arr)):
        if j > BED: # if no point is found before BED return false
            return False 
        if pxVal < 30: # if a point is found return that points y value
            return j

def getContactAngle(img, img_T, name):
    # More efficient to convert this to a np array later
    xLeft = []
    yLeft = []
    xRight = []
    yRight = []

    syringeFound = False
    done = False
    
    for i,pxVal in enumerate(np.nditer(img[0])):

        # Condition for finding left side of syringe
        if pxVal < 30 and syringeFound == False: 

            # Look at every point distance 2 from syringe to distance 100
            for k in xrange(i-5,i-100,-1):
                j = findEdge(img_T[k])
                if j:
                    xLeft.append(k) # column or x
                    yLeft.append(230-j) # row or y
                else:
                    break # if no pixel found before BED break

            syringeFound = True # Raise flag

        # Condition for finding right side of syringe
        if pxVal > 220 and syringeFound == True:

            # Look at every point distance 2 from syringe to distance 100
            for k in xrange(i+5,i+100,1):
                j = findEdge(img_T[k])
                if j:
                    xRight.append(k) # column or x
                    yRight.append(230-j) # row or y
                else:
                    break # if no pixel found before BED break        

            done = True

        # If both edges of the syringe have been read you're done
        if done: 
            break
            
    ###
    ### Fit left side
    ###

    # Convert to np.arrays
    xLeft = np.asarray(xLeft)
    yLeft = np.asarray(yLeft)

    # Fit Left side data set with 3rd degree polynomial
    coefLeft = np.polyfit(xLeft,yLeft,3)
    polyLeft = np.poly1d(coefLeft)

    # Get values to plot fit, assume min and max locations
    xsLeft = arange(xLeft[-1], xLeft[0], 0.1)
    ysLeft = polyLeft(xsLeft)

    # Get derivate @ BED and values to plot dervative
    derivLeft = polyLeft.deriv()
    mL = derivLeft(xLeft[-1])
    bL = yLeft[-1] - mL*xLeft[-1]
    x1L,y1L = (-1000, mL*-1000+bL)
    x2L,y2L = (1000, mL*1000+bL)

    ###
    ### Fit right side
    ###

    # Convert to np.arrays
    xRight = np.asarray(xRight)
    yRight = np.asarray(yRight)

    # Fit Right side data set with 3rd degree polynomial
    coefRight = np.polyfit(xRight,yRight,3)
    polyRight = np.poly1d(coefRight)

    # Get values to plot fit, assume min and max locations
    xsRight = arange(xRight[0], xRight[-1], 0.1)
    ysRight = polyRight(xsRight)

    # Get derivate @ BED and values to plot dervative
    derivRight = polyRight.deriv()
    mR = derivRight(xRight[-1])
    bR = yRight[-1] - mR*xRight[-1]
    x1R,y1R = (-1000, mR*-1000+bR)
    x2R,y2R = (1000, mR*1000+bR)
    
    # Plot Left lines
    plot((x1L,x2L),(y1L,y2L),'b')
    plot(xLeft,yLeft,'r.')
    plot(xsLeft, ysLeft, 'g')
    
    # Plot Right Lines
    plot((x1R,x2R),(y1R,y2R),'b')
    plot(xRight,yRight,'r.')
    plot(xsRight, ysRight, 'g')
    axhline(230-BED)
    xlim(95,275)
    ylim(140,240)
    
    fileName = name + "-plt.png"
    
    savefig(fileName, dpi=150, papertype='b10', orientation='portrait', bbox_inches='tight')
    
    clf() # Clear figure
    
    return np.arctan(mL)*180/np.pi, np.arctan(-1*mR)*180/np.pi

BED = 85 # Bubble edge depth as set by user
inputPath = "D:\\GitHub\\workspace\\A2_work\\work\\"

leftAngle = []
rightAngle = []
count = []

for (root, subFolders, files) in os.walk(inputPath):

    #Second condition forces this to run in only the index dir
    for item in files:
        
        if '.png' in item and 'plt' not in item:
            img2 = cv2.imread(root+'\\'+item)
            img = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) # convert to gray
            
            leftCA, rightCA = getContactAngle(img, img.T, root+'\\'+item[:-4])
            
            leftAngle.append(leftCA)           
            rightAngle.append(rightCA)
            count.append(int(item[:-4]))
            print item, leftCA, rightCA
        
        if '0660' in item:
            break

# Plot left CA and right CA vs image number and add a legend and axis labels
plot(count,leftAngle,'r')
plot(count,rightAngle,'b')
ylabel('ContactAngle (Degrees)', size=8)
xlabel('Image Number', size=8)
title('TEST001 Contact Angle vs Image Number', size=8)
p1 = plt.Rectangle((0, 0), 1, 1, fc='r')
p2 = plt.Rectangle((0, 0), 1, 1, fc='b')
legend([p1,p2], ['Left CA','Right CA'], loc=1,prop={'size':8})
            
savefig(inputPath+'Summary.png', dpi=150, papertype='b10', orientation='portrait', bbox_inches='tight')#, format='pdf')


def videoToFilteredVideo(videoPath, outPath, N, deg, roi, debug=False):
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
            
            # Convert back to BGR for video codec
            img_Color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            
            if debug:
                # Convert Nom to grayscale so it can output in the same png
                # Then save image as side by side before / after png
                img_Nom = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) 
                img_Debug = np.hstack(img,img_Nom)        
                cv2.imwrite(outPath+imgNum+'.png', img_Debug)
            else:
                cv2.imwrite(outPath+imgNum+'.png', img_Color)
 
            video.write(img_Color)
            
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
inputVid="D:\\GitHub\\workspace\\A2_work\\20140728-14_30_42-CA1_M01SS-27.MOV"
outputPath="D:\\GitHub\\workspace\\A2_work\\work\\"
rotation = 2 # degrees
roi = (850,1080,800,1175)  # xmin:xmax (vert, row), ymin:ymax (horz, col)
N = 10
print inputVid
print outputPath
videoToFilteredVideo(inputVid, outputPath, N, rotation, roi)

if __name__ == '__main__':
    default = dict(
    PATH = 'D:\\GitHub\\workspace\\A2_work\\20140728-14_30_42-CA1_M01SS-27.MOV',
    VSF = 80,
    VEF = 660,
    N = 10,
    R = 2,
    ROI = (850,1080,800,1175), # row_min, row_max, col_min, col_max
    MASK = (90,280,255), # col_min, col_max, value outside mask
    BR = 90,
    TH = 160,
    BED = 150, # row (0 is at top of image)
)


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
