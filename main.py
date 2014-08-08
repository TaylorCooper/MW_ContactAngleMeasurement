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
                        
    Outputs Normal Mode:
        .txt containing frameNumber, leftCA, rightCA, radius
    Outputs Debug Mode:
        .png comparing filtered and unfiltered image
        .png of fits and points fitted
        .txt containing frameNumber, leftCA, rightCA, radius
    Dependencies:
        numpy, cv2, os, string, pylab, shutil, csv
    Limitations:
        Cannot measure contact angles > 90 degrees
    
    Pending Major Changes:
        Tuning: CA1
                08 video inadequate
                07,21, 34, 38 increase width 
                11,15,34 increase video length threshold? 
                17 bad bed / noisy
                20 noisy
                27 move mask in maybe...? region C is problematic
                
                CA2
                04, 08, 21,33,38 width
                34 incorrect bed
                46 noisy
                40,49 video length
                
                CA3
                1,2,3,6,8,10,11,12,13,34,44 width
                22.40,41,42,43,46 goes over 90
                31,32,56 video length
                40 N resolution may not be fine enough to see receding CA
        
        Rename data.png to filename / make it first file
        Overlay plots on image
        Save bed selection image?
        Faster way to find middle image...
        Run just get bed on the whole folder then do analysis
        Measurement algo for tilt goniometer photos or side injection
        Way to measure angles over 60 or 90... seems to have problems right now
        
    History:                  
    --------------------------------------------------------------
    Date:    
    Author:    Taylor Cooper
    Modification:    
     --------------------------------------------------------------
"""

import numpy as np
import cv2, os, string, shutil, csv
#from matplotlib.pyplot import *
from pylab import *
import config

MAX_VIDEO_SIZE = 1000 #Frames

# Config file variables
vars = {
    # Always provide the trailing \\ for 'PATH' if it's a filepath
    'PATH' : 'D:\\GitHub\\AFCC Metal FSU Testing 2014\\A2\\',
    'VSF' : 50,
    'VEF' : 1500,
    'N' : 10,
    'R' : 2,
    'ROI' : (750,1080,800,1175), # row_min, row_max, col_min, col_max
    'MASK' : (90,280,255), # col_min, col_max, value outside mask
    'BR' : 90,
    'TH' : 180,
    'DPI' : 100, # plot quality (lower = faster
    'DEBUG' : True,
    'RESETBED' : False
}

class getContactAngle():
    """
    Description: Gets contact angle from folder of goniometer videos.
    Input: .MOV
    Output: Filtered and contact angle plots
    """
    
    def __init__(self, configurables):
        """ Initialize getContactAngle.
        Input:  configFile parameters
        Output:  None
        """
        
        # Configurable parameters
        self.path = configurables['PATH']
        self.vsf = configurables['VSF']
        self.vef = configurables['VEF']
        self.n = configurables['N']
        self.r = configurables['R']
        self.roi = configurables['ROI']
        self.mask = configurables['MASK']
        self.br = configurables['BR']
        self.th = configurables['TH']
        self.dpi = configurables['DPI']

        self.debug = configurables['DEBUG']
        self.resetbed = configurables['RESETBED']
        
        # Internal parameters
        self.workDir = None
        self.vidPath = None
        self.sygOffset = 6
        self.bubMax = 200
        self.minFitPoints = 4
        self.w = self.roi[3] - self.roi[2]
        self.h = self.roi[1] - self.roi[0]
        
        # Human input parameters, warning this value changes with the roi
        self.bed = None

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
    
        makePlot = False
        caL = None # left contact angle
        caR = None # right contact angle
    
        # Condition to make sure polyfit will not crash    
        if xLeft != [] and len(xLeft) > self.minFitPoints:
            makePlot = True

            # Convert to np.arrays
            xLeft = np.asarray(xLeft)
            yLeft = np.asarray(yLeft)
        
            # Fit data set with 3rd degree polynomial
            coefLeft = np.polyfit(xLeft,yLeft,3)
            polyLeft = np.poly1d(coefLeft)
               
            # Get values to plot fit, assume min and max locations
            xsLeft = np.arange(xLeft[-1], xLeft[0], 0.1)
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

        if xRight != [] and len(xRight) > self.minFitPoints:
            makePlot = True

            # Same as above
            xRight = np.asarray(xRight)
            yRight = np.asarray(yRight)
            coefRight = np.polyfit(xRight,yRight,3)
            polyRight = np.poly1d(coefRight)
            xsRight = np.arange(xRight[0], xRight[-1], 0.1)
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

        # Only generate plots in debug mode
        if makePlot and self.debug:
            axhline(self.h-self.bed)
            fileName = self.workDir + "plt-" + count + ".png"
            
            xlim(self.w/2-100,self.h/2+100) # 87-287
            ylim(self.h-self.bed-10,self.h+10) # 135-240
            title(fileName, size=8)
    
            savefig(fileName, dpi=self.dpi, papertype='b10', 
                    orientation='portrait', bbox_inches='tight')
        
            clf() # Clear figure
        
        if xLeft != [] and xRight != []:
            radius = max(xRight) - min(xLeft)
        elif xLeft != []:
            # Radius with estimated syringe width
            radius = max(xLeft) - min(xLeft) + self.sygOffset*2 + 4
        elif xRight != []:
            # Radius with estimated syringe width
            radius = max(xRight) - min(xRight) + self.sygOffset*2 + 4
        else:
            radius = None
        
        return radius,caL,caR 



    def findSyringeEdge(self, img, img_T, count):
        """Finds left then right edge of syringe needle at top of image.
        
        Input: np.array of img and transposed np.array of img, frame number
        
        Output: Returns left, right contact angles
        """

        xLeft = [] # Column positions on bubble left of syringe
        yLeft = [] # Matching row positions on bubble left of syringe
        xRight = []
        yRight = []
    
        syringeFound = False
        done = False
        
        for i,pxVal in enumerate(np.nditer(img[0])):
    
            # Condition for finding left side of syringe
            if pxVal < 30 and syringeFound == False: 
    
                # Look at every point distance 3 from syringe to distance 100
                for k in xrange(i-self.sygOffset,i-self.bubMax,-1):
                    j = self.findBubbleEdge(img_T[k])
                    if j:
                        xLeft.append(k) # column or x
                        yLeft.append(self.h-j) # row or y
                    else:
                        break # if no pixel found before BED break
                    
                    if k == i-self.bubMax+10:
                        print 'Warning:: Bubble may exceed min left px', k
    
                syringeFound = True # Raise flag
    
            # Condition for finding right side of syringe
            if pxVal > 220 and syringeFound == True:
    
                # Look at every point distance 3 from syringe to distance 100
                for k in xrange(i+self.sygOffset,i+self.bubMax,1):
                    j = self.findBubbleEdge(img_T[k])
                    if j:
                        xRight.append(k) # column or x
                        yRight.append(self.h-j) # row or y
                    else:
                        break # if no pixel found before BED break        

                    if k == i+self.bubMax-10:
                        print 'Warning:: Bubble may exceed max right px', k
    
                done = True
    
            # If both edges of the syringe have been read you're done
            if done: 
                break
        
        # Return contact angles or none if no data points found (caL,caR)   
        return self.fitContactAngle(xLeft, yLeft, xRight, yRight, count)


    # mouse callback function
    def filterImg(self, img, rot_Matrx, imgNum):
        """Filters and saves every Nth image in a video with getContactAngle()
        
        Input:
            self.vidPath = a string, path of video to be analyzed
            self.workDir = a string, working directory
            Parameters from config.py
        
        Output:
            A cropped, filtered, rotated .png (in debug mode)
            A call to find syringe edge
            A csv log file
        """
        
        rows,cols,channels = img.shape
        
        # Rotate and crop image based on self.r / self.roi
        img = cv2.warpAffine(img,rot_Matrx,(cols,rows))
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
        
        if self.debug: # Save debugging image
            img_Nom = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY) 
            img_Debug = np.hstack((img,img_Nom))     
            cv2.imwrite(self.workDir+imgNum+'.png', img_Debug)
        
        return img

    def measureVideo(self):
        """Filters and saves every Nth image in a video with getContactAngle()
        
        Input:
            self.vidPath = a string, path of video to be analyzed
            self.workDir = a string, working directory
            Parameters from config.py
        
        Output:
            A cropped, filtered, rotated .png (in debug mode)
            A call to find syringe edge
            A csv log file
        """
        
        count = 0
        leftAngles = []
        rightAngles = []
        leftFrames = []
        rightFrames = []
        radii = []
        radiusFrames = []
        makePlot = False
        
        # Start log file
        logPath = self.workDir + 'data.csv'
        logFile = open(logPath, 'wb')
        log = csv.writer(logFile, delimiter=',', quoting=csv.QUOTE_NONE)
        log.writerow(['frameNumber','radius','leftCA','rightCA']) # Header
        
        # Rotation matrix centered in self.roi and rotated by self.r degrees
        x_cent = int((self.roi[0]+self.roi[1])/2)
        y_cent = int((self.roi[2]+self.roi[3])/2)
        rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),self.r,1)
        
        # Start reading input video
        cap = cv2.VideoCapture(self.vidPath)
            
        while True:
            ret, frame = cap.read()
            
            # If frameNum/N=0 is returned do analysis
            if ret and (count % self.n) == 0 and count >= self.vsf:          
                            
                imgNum = string.zfill(count,4) # Change 1 to 0001
                
                # Filter image
                img = self.filterImg(frame, rot_Matrx, imgNum)
                
                # Get contact angles and plot results 
                radius, caL, caR = self.findSyringeEdge(img, img.T, imgNum)

                if self.debug:           
                    print imgNum, radius, caL, caR
                
                # Append to lists for plots later
                if caL:
                    leftAngles.append(caL)
                    leftFrames.append(count)
                if caR:
                    rightAngles.append(caR)
                    rightFrames.append(count)
                if radius:
                    radii.append(radius)
                    radiusFrames.append(count)
                
                # Output data to log file
                log.writerow([count,radius,caL,caR])
                     
            if count > self.vef or not ret:
                cap.release()
                break
            
            count += 1

        logFile.close() # Close your log file

        # Plot 
        if radii != []:
            makePlot = True
            plot(radiusFrames, radii, 'g')
        
        if leftAngles != []:
            makePlot = True
            plot(leftFrames,leftAngles,'r')
            
        if rightAngles != []:
            makePlot = True
            plot(rightFrames,rightAngles,'b')
            
        if makePlot:
            ylabel('ContactAngle (Degrees), Radius (Px)', size=8)
            xlabel('Image Number', size=8)
            title('TEST001 Contact Angle & Radius vs Image Number', size=8)
            p1 = plt.Rectangle((0, 0), 1, 1, fc='g')
            p2 = plt.Rectangle((0, 0), 1, 1, fc='r')
            p3 = plt.Rectangle((0, 0), 1, 1, fc='b')
            legend([p1,p2,p3], ['radius','leftCA','rightCA'], 
                   loc=1,prop={'size':8})
                    
            savefig(self.workDir+'data.png', dpi=400, papertype='b10', 
                    orientation='portrait', bbox_inches='tight')
            clf() # Clear your figure
    
    def mouseCallback(self,event,x,y,flags,param):
        """ Mouse callback function, selects cursor point to populate self.bed
        """
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bed = y
            print 'You selected: ', x, y
            
    def userSelectBed(self, img):
        """ Opens window and prompts user to select BED     
        Input: Filtered image.
        Output: None
        """
               
        # Set up window & call back
        windowName = 'Left click base of bubble. Press Esc to exit.'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL) # Can be resized
        cv2.resizeWindow(windowName, self.w*3, self.h*3) # Resize window
        cv2.setMouseCallback(windowName, self.mouseCallback) # Set callback
        
        imgDefault = img # Ideally this resets the image later
        
        while True:
            # Plot cursor click
            if self.bed:
                h = self.bed
            else:
                h = 0
            cv2.line(img, (-1000,h), (1000,h), (0,0,0))
            cv2.imshow(windowName, img)
            img = imgDefault
            k = cv2.waitKey(1000) & 0xFF # Basically delay 1000 ms
            print 'BED is currently: ', self.bed
            if k == 27: # Press escape to exit
                break
            
        cv2.destroyAllWindows()
        
    def getBED(self):
        """Attempts to pull BED from file name else prompts user to select it.
        
        Input: videoPath
        Output: renames video file to store BED
        """
        
        count = 0
        
        # Pull BED from file name assumes this file structure:
        # [Name of any format exclude the string _BED]_BED###.MOV
        # ### indicating an integer value for BED counted down from the top
        # of the cropped and filtered image.
        if '_BED' in self.vidPath and self.resetbed == False:
            self.bed = int(self.vidPath.split('BED')[1][:-4])
            return
        
        # Manually get BED, let's choose frame 250 randomly for this
        
        # Rotation matrix centered in self.roi and rotated by self.r degrees
        x_cent = int((self.roi[0]+self.roi[1])/2)
        y_cent = int((self.roi[2]+self.roi[3])/2)
        rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),self.r,1)
        
        # Start reading input video
        cap = cv2.VideoCapture(self.vidPath)
        
        # Really inefficient way to find video mid point
        while True:
            ret, frame = cap.read()
            if count > self.vef or not ret:
                cap.release()
                break
            count += 1
        
        # Now call userSelectedBED on correct frame
        midPoint = int(count/2)
        count = 0
        cap = cv2.VideoCapture(self.vidPath)
        
        # User selects BED from filtered image
        while True:
            ret, frame = cap.read()
            # If frameNum/N=0 is returned do analysis
            
            if count == midPoint:          
                imgNum = string.zfill(midPoint,4)
                # Filter image and get BED
                img = self.filterImg(frame, rot_Matrx, imgNum)
                self.userSelectBed(img)
            
            if count > self.vef or not ret:
                cap.release()
                break
            
            count += 1
         
        # Rename file so this doesn't need to happen again
        if self.resetbed: # Remove _BED###.MOV
            bedName = self.vidPath[:-11]+'_BED'+str(self.bed)+'.MOV'
        else: # Remove .MOV
            bedName = self.vidPath[:-4]+'_BED'+str(self.bed)+'.MOV'
        print 'Renaming: ', self.vidPath
        print 'New Name: ', bedName
        os.rename(self.vidPath, bedName)
        self.vidPath = bedName
        
    
    def run(self):
        """ Executes getContactAngle on an individual video or recursively over 
        an entire folder tree. 
        
        Input:
        Parameters from config.py
        
        Output:
        In non debug mode it deletes folders with duplicate names.
        """
        
        if not self.debug:
            imgMS = cv2.imread('D:\\GitHub\\workspace\\A2_ContactAngle\\\
                                welcome.png')
            cv2.imshow('image',imgMS)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        if '.MOV' in self.path: # For individual file call directly
            self.workDir = self.path[:-4]+'\\'
            self.vidPath = self.path
            print 'Analyzing: ', self.vidPath
            
            self.getBED() # Pull BED from filename or ask user
            
            # If self.workDir exists and not running debug mode, delete it
            if os.path.isdir(self.workDir)and not self.debug:
                print 'Deleting: ', self.workDir
                shutil.rmtree(self.workDir)
            
            # Make self.workDir if it doesn't exist
            if not os.path.isdir(self.workDir):
                print 'Making: ', self.workDir
                os.mkdir(self.workDir)
            
            # Measure contact angles in Video and record outputs to workDir
            self.measureVideo()
            
        else: # Directory given, use os.walk
            for (root, subFolders, files) in os.walk(self.path):
                #Second condition forces this to run in only the index dir
                for item in files:
                    if '.MOV' in item:
                        
                        if 'FIX' in item: continue # Ignore pre-filtered video
                        
                        self.vidPath = root+'\\'+item
                        self.workDir = root+'\\'+item[:-4]+'\\'
                        print 'Analyzing: ', self.vidPath
                        
                        self.getBED() # Pull BED from filename or ask user
                        
                        if os.path.isdir(self.workDir) and not self.debug:
                            print 'Deleting: ', self.workDir
                            shutil.rmtree(self.workDir)
                        
                        if not os.path.isdir(self.workDir):
                            print 'Making: ', self.workDir
                            os.mkdir(self.workDir)
                        
                        self.measureVideo()
    


if __name__ == '__main__':
    gCA = getContactAngle(vars)
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
