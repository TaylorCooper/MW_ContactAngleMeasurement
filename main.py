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

# Config file variables
vars = {
    'PATH' : 'D:\\GitHub\\workspace\\A2_work\\SampleAnalysis\\',
    'VSF' : 80,
    'VEF' : 660,
    'N' : 10,
    'R' : 2,
    'ROI' : (850,1080,800,1175), # row_min, row_max, col_min, col_max
    'MASK' : (90,280,255), # col_min, col_max, value outside mask
    'BR' : 90,
    'TH' : 160,
    'BED' : 85, # row (0 is at top of image)
    'DPI' : 100, # plot quality (lower = faster
    'W' : vars['ROI'][3]-vars['ROI'][2],
    'H' : vars['ROI'][1]-vars['ROI'][0]
}

class getContactAngle():
    """
    Description: Gets contact angle from folder of goniometer videos.
    Input: .MOV
    Output: Filtered and contact angle plots
    """
    
    def __init__(self, vars, debug=True):
        """ Initialize getContactAngle.
        Input:  configFile parameters
        Output:  None
        """
        self.path = vars['PATH']
        self.vsf = vars['VSF']
        self.vef = vars['VEF']
        self.n = vars['N']
        self.r = vars['R']
        self.roi = vars['ROI']
        self.mask = vars['MASK']
        self.br = vars['BR']
        self.th = vars['th']
        self.bed = vars['BED']
        self.dpi = vars['dpi']
        self.w = vars['W']
        self.h = vars['H']
        self.debug = debug

    def findBubbleEdge(self, arr):
        """ Find the top edge of a goniometer bubble.
        Input:  Transposed numpy array of filtered image.
        Output:  Returns false if no edge found, else edge position.
        """
        # j is position in arr, pxVal is value of the pixel
        for j,pxVal in enumerate(np.nditer(arr)):
            if j > self.bed: # if no point is found before BED return false
                return False 
            if pxVal < 30: # if a point is found return that points y value
                return j
                            
    
    def fitContactAngle(self, xLeft, yLeft, xRight, yRight, count):
        """Plots left, right contact angles based on 3rd degree polyfit. 
        
        Input: 4 lists to plot and a frame for naming
        
        Output: saves .png of data points, fits, contact angles and self.BED
        Returns left, right contact angle or none.
        """
    
        plot = False
        caL = None # left contact angle
        caR = None # right contact angle
    
        # Condition to make sure polyfit will not crash    
        if xLeft != [] and yLeft != [] and len(xLeft) == len(yLeft):
            plot = True

            # Convert to np.arrays
            xLeft = np.asarray(xLeft)
            yLeft = np.asarray(yLeft)
        
            # Fit data set with 3rd degree polynomial
            coefLeft = np.polyfit(xLeft,yLeft,3)
            polyLeft = np.poly1d(coefLeft)
               
            # Get values to plot fit, assume min and max locations
            xsLeft = arange(xLeft[-1], xLeft[0], 0.1)
            ysLeft = polyLeft(xsLeft)
        
            # Get derivative @ left most point and values to plot derivative
            derivLeft = polyLeft.deriv()
            mL = derivLeft(xLeft[-1])
            
            if self.debug:
                # Get CA lines to plot
                bL = yLeft[-1] - mL*xLeft[-1]
                x1L,y1L = (-1000, mL*-1000+bL)
                x2L,y2L = (1000, mL*1000+bL)
                
                # Plot lines
                plot((x1L,x2L),(y1L,y2L),'b')
                plot(xLeft,yLeft,'r.')
                plot(xsLeft, ysLeft, 'g')

            caL = np.arctan(mL)*180/np.pi

        if xRight != [] and yRight != [] and len(xRight) == len(yRight):
            plot = True

            # Same as above
            xRight = np.asarray(xRight)
            yRight = np.asarray(yRight)
            coefRight = np.polyfit(xRight,yRight,3)
            polyRight = np.poly1d(coefRight)
            xsRight = arange(xRight[0], xRight[-1], 0.1)
            ysRight = polyRight(xsRight)
            derivRight = polyRight.deriv()
            mR = derivRight(xRight[-1])
            
            if self.debug:
                bR = yRight[-1] - mR*xRight[-1]
                x1R,y1R = (-1000, mR*-1000+bR)
                x2R,y2R = (1000, mR*1000+bR)
                plot((x1R,x2R),(y1R,y2R),'b')
                plot(xRight,yRight,'r.')
                plot(xsRight, ysRight, 'g')
        
            caR = np.arctan(-1*mR)*180/np.pi

        if plot and self.debug:
            axhline(self.h-self.bed)
            fileName = self.path + "plt-" + count + ".png"
            
            xlim(self.w/2-100,self.h/2+100) # 87-287
            ylim(self.h-self.bed-10,self.h+10) # 135-240
            title(fileName, size=8)
    
            savefig(fileName, dpi=self.dpi, papertype='b10', 
                    orientation='portrait', bbox_inches='tight')
        
            clf() # Clear figure
        
        return caL,caR 



    def findSyringeEdge(self, img, img_T, count):
        """Finds left then right edge of syringe needle at top of image.
        
        Input: np.array of img and transposed np.array of img, frame number
        
        Output: Returns left, right contact angles
        """
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
    
                # Look at every point distance 3 from syringe to distance 100
                for k in xrange(i-3,i-100,-1):
                    j = self.findBubbleEdge(img_T[k])
                    if j:
                        xLeft.append(k) # column or x
                        yLeft.append(self.h-j) # row or y
                    else:
                        break # if no pixel found before BED break
    
                syringeFound = True # Raise flag
    
            # Condition for finding right side of syringe
            if pxVal > 220 and syringeFound == True:
    
                # Look at every point distance 3 from syringe to distance 100
                for k in xrange(i+3,i+100,1):
                    j = self.findBubbleEdge(img_T[k])
                    if j:
                        xRight.append(k) # column or x
                        yRight.append(self.h-j) # row or y
                    else:
                        break # if no pixel found before BED break        
    
                done = True
    
            # If both edges of the syringe have been read you're done
            if done: 
                break
        
        # Return contact angles or none if no data points found (caL,caR)   
        return self.fitContactAngle(xLeft, yLeft, xRight, yRight, count)


    def videoToFilteredImgs(self, videoPath, workDir):
        """Filters and saves every Nth image in a video with getContactAngle()
        
        Input:
        videoPath = a string, path of video to be analyzed
        outPath = a string, working directory
        N = a positive integer, images divisable by it will be analyzed
        
        Output:
        A cropped, filtered, rotated image
        """
        
        leftAngles = []
        rightAngles = []
        leftFrames = []
        rightFrames = []
        plot = False
        count = 0
        
        # Rotation matrix centered in self.roi and rotated by self.r degrees
        x_cent = int((self.roi[0]+self.roi[1])/2)
        y_cent = int((self.roi[2]+self.roi[3])/2)
        rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),self.r,1)
        
        # Start reading input video
        cap = cv2.VideoCapture(videoPath)
            
        while True:
            ret, frame = cap.read()
            
            # If frameNum/N=0 is returned do analysis
            if ret and (count % self.n) == 0 and count >= self.vsf:          
                            
                imgNum = string.zfill(count,4) # Change 1 to 0001
                
                rows,cols,channels = frame.shape
                
                # Rotate and crop image based on self.r / self.roi
                img = cv2.warpAffine(frame,rot_Matrx,(cols,rows))
                # Crop to self.roi [row_min:row_max, col_min:col+max] 
                img_Nom = img[self.roi[0]:self.roi[1],self.roi[2]:self.roi[3]]           
                
                # Gray scale and apply a mask to remove unnecessary points
                img = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) # convert to GS
                mask = np.zeros([self.h,self.w], np.uint8)
                mask[:,:self.mask[0]] = self.mask[2]
                mask[:,self.mask[1]:] = self.mask[2]
                img = cv2.add(img,mask)
    
                # Brighten all pixels then apply binary threshold
                img = cv2.add(img,self.br)
                r,img = cv2.threshold(img,self.th,255,cv2.THRESH_BINARY)
                
                # Get contact angles and plot results 
                caL, caR = self.findSyringeEdge(img, img.T, imgNum)

                if self.debug:           
                    print imgNum, caL, caR
                
                if caL:
                    leftAngles.append(caL)
                    leftFrames.append(count)
                if caR:
                    rightAngles.append(caR)
                    rightFrames.append(count)
                
                if self.debug:
                    # Convert Nom to grayscale so it can output in the same png
                    # Then save image as side by side before / after png
                    img_Nom = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) 
                    img_Debug = np.hstack(img,img_Nom)        
                    cv2.imwrite(workDir+imgNum+'.png', img_Debug)
                     
            if count > self.vef or ret == False:
                print count
                break
            
            count += 1
    
        # Plot left CA and right CA vs image number and add a legend and axis labels
        if leftAngles != []:
            plot = True
            plot(leftFrames,leftAngles,'r')
        if rightAngles != []:
            plot = True
            plot(rightFrames,rightAngles,'b')
        if plot:
            ylabel('ContactAngle (Degrees)', size=8)
            xlabel('Image Number', size=8)
            title('TEST001 Contact Angle vs Image Number', size=8)
            p1 = plt.Rectangle((0, 0), 1, 1, fc='r')
            p2 = plt.Rectangle((0, 0), 1, 1, fc='b')
            legend([p1,p2], ['Left CA','Right CA'], loc=1,prop={'size':8})
                    
            savefig(videoPath[:-4]+'-Summary.png', dpi=self.dpi, 
                    papertype='b10', orientation='portrait', 
                    bbox_inches='tight')
        
            clf()
        
    
    def run(self):
        """ Use os.walk to do stuff
        """
        

        
        if '.MOV' in self.path: # For individual file call directly
            workDir = self.path[:-4]
            os.mkdir(workDir)
            self.videoToFilteredImgs(self.path, workDir)
        else:
            for (root, subFolders, files) in os.walk(self.path):
                #Second condition forces this to run in only the index dir
                for item in files:
                    print item, root            
                    if '.MOV' in item:
                        path = root+'\\'+item
                        workDir = path[:-4]
                        os.mkdir(workDir)
                        self.videoToFilteredImgs(path, workDir)
    


if __name__ == '__main__':
    gCA = getContactAngle(vars, debug=True)
    gCA.run()

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
