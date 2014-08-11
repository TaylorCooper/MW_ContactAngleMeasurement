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
        .txt containing frameNumber, leftCA, rightCA, diameter
    Outputs Debug Mode:
        .png comparing filtered and unfiltered image
        .png of fits and points fitted
        .txt containing frameNumber, leftCA, rightCA, diameter
    Dependencies:
        numpy, cv2, os, string, pylab, shutil, csv, time
    Limitations:
        Cannot measure contact angles > 90 degrees
    
    Pending Major Changes:
        Tuning: CA1
                08 video inadequate
                17 bad bed / noisy
                20 noisy
                27 move mask in maybe...? region C is problematic
                
                CA2
                46 noisy
                
                CA3
                40 N resolution may not be fine enough to see receding CA
                
                
        Tilting Changes:
            Changed polyfit to 3degree
            Changed bubmax to 600
            Changed derivOffset to 20... (will need to change on a per plot
            basis for this one..)
        
        Overlay plots on image
        Consider storing syringe loc as a self var or have a mod to pick it
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
import cv2, os, string, shutil, csv, time, sys
from pylab import * # plotting tools
import config

# Video config file variables
varsVid = {
    # Always provide the trailing \\ for 'PATH' if it's a filepath
    'PATH' : 'D:\\GitHub\\AFCC Metal FSU Testing 2014\\20140808_Dataset_CA1\\',
    'VSF' : 100,
    'VEF' : 1500,
    'N' : 5,
    'R' : 2,
    'ROI' : (750,1080,800,1175), # row_min, row_max, col_min, col_max
    'MASK' : (30,340,255), # col_min, col_max, value outside mask
    'BR' : 90,
    'TH' : 180,
    'POLYDEG' : 5,
    'DERIVOFF' : 10,
    'DPI' : 100, # plot quality (lower = faster
    'DEBUG' : True, # Plots and saves images
    'RESETBED' : False, # Will ignore _BED if present
    'ASSIGNBEDS' : False, # Will ignore analysis and assign beds, responds to RESETBED
    'REMOVEBEDS' : False, # Will ignore assignBeds and analysis and remove _BED
}

headersVid = [
          'Filename',
          'FolderID',
          'FabID',
          'Sample#',
          'Region',
          'AdvCALeft(deg)',
          'AdvCARight(deg)',
          'MaxDiameter(px)',
          'BED(px)'       
          ]

# Tilt Angle config file variables
varsImg = {
    # Always provide the trailing \\ for 'PATH' if it's a filepath
    'PATH' : 'D:\\_Work\\0000-AFCC_MetalFSU_A2\\20140810_TiltPhotos\\Derv10_Poly8\\',
    'R' : 2,
    'BR' : 130,
    'TH' : 190,
    'POLYDEG' : 8,
    'DERIVOFF' : 10, # Or 20 in some cases    
    'DPI' : 100, # plot quality (lower = faster
    'DEBUG' : True, # Plots and saves images
    'RESETBED' : False, # Will ignore _BED if present
    'ASSIGNBEDS' : False, # Will ignore analysis and assign beds, responds to RESETBED
    'REMOVEBEDS' : False, # Will ignore assignBeds and analysis and remove _BED
}

headersImg = [
          'Filename',
          'Material',
          'index',
          'tiltAngle(deg)',
          'RecedingCALeft(deg)',
          'AdvCARight(deg)',
          'MaxDiameter(px)',
          'BED(px)',
          ]


class getContactAngle():
    """
    Description: Gets contact angle from folder of goniometer videos.
    Input: .MOV
    Output: Filtered and contact angle plots
    """
    
    def __init__(self, configurables, masterLogHeaders, assignBeds=False, 
                 isVideo=True):
        """ Initialize getContactAngle.
        Input:  configFile parameters
        Output:  None
        """
        
        self.isVideo = isVideo
        
        # Video only parameters
        if isVideo:
            self.vsf = configurables['VSF']
            self.vef = configurables['VEF']
            self.n = configurables['N']
            self.roi = configurables['ROI']
            self.mask = configurables['MASK']
            
            # Internal Parameters
            self.w = self.roi[3] - self.roi[2]
            self.h = self.roi[1] - self.roi[0]
        else:
            self.w = None
            self.h = None

            
        self.path = configurables['PATH']
        self.r = configurables['R']
        self.br = configurables['BR']
        self.th = configurables['TH']
        self.polyDeg = configurables['POLYDEG'] # Default 5 for videos
        self.derivOffset = configurables['DERIVOFF'] # Default 10 for videos     
        self.dpi = configurables['DPI']
        self.debug = configurables['DEBUG']
        self.resetbed = configurables['RESETBED']
        self.assignBeds = configurables['ASSIGNBEDS']
        self.removeBeds = configurables['REMOVEBEDS']
        self.headers = masterLogHeaders
        
        # Internal parameters
        self.workDir = None
        self.itemPath = None
        self.syringMinPos = 120
        self.sygOffset = 6 
        self.bubMax = 600
        self.minFitPoints = 12

        
        # Human input parameters, warning this value changes with the roi
        self.bed = None
                         
    
    def fitContactAngle(self, xLeft, yLeft, xRight, yRight, count):
        """Plots left, right contact angles based on polyfit.
        
        Input: 4 lists to plot and a frame for naming
        
        Output: saves .png of data points, fits, contact angles and self.BED
        Returns left, right contact angle or none.
        """
    
        # Warning: xLeft = xFromLeft etc. in this function
        makePlot = False
        caL = None # left contact angle
        caR = None # right contact angle
    
        # Condition to make sure polyfit will not crash    
        if xLeft != [] and len(xLeft) > self.minFitPoints:
            makePlot = True

            # Convert to np.arrays
            xLeft = np.asarray(xLeft)
            yLeft = np.asarray(yLeft)
        
            # Fit data set with self.polyDeg polynomial
            coefLeft = np.polyfit(xLeft,yLeft,self.polyDeg)
            polyLeft = np.poly1d(coefLeft)
               
            # Get values to plot fit, assume min and max locations
            xsLeft = np.arange(xLeft[0], xLeft[-1], 0.1)
            ysLeft = polyLeft(xsLeft)
        
            # Get derivative @ left most point and values to plot derivative
            # Derivative calculated up from edge, images too noisy for
            # calculation right at self.bed
            derivLeft = polyLeft.deriv()
            # Invert because rotating graph
            mL = 1/derivLeft(xLeft[0]+self.derivOffset) 

            
            if self.debug:
                # Get CA lines to plot
                bL = xLeft[0] - mL*yLeft[0] # Also inverted
                x1L,y1L = (-1000, mL*-1000+bL)
                x2L,y2L = (1000, mL*1000+bL)
                
                # Plot lines
                plot((x1L,x2L),(y1L,y2L),'b')
                plot(yLeft, xLeft,'r.') #Inverted
                plot(ysLeft, xsLeft, 'g') #Inverted

            # Get contact angle
            caL = np.arctan(mL)*180/np.pi
            if caL < 0: caL = 180+caL

        if xRight != [] and len(xRight) > self.minFitPoints:
            makePlot = True

            # Same as above
            xRight = np.asarray(xRight)
            yRight = np.asarray(yRight)
            coefRight = np.polyfit(xRight,yRight,self.polyDeg)
            polyRight = np.poly1d(coefRight)
            xsRight = np.arange(xRight[0], xRight[-1], 0.1)
            ysRight = polyRight(xsRight)
            derivRight = polyRight.deriv()
            mR = 1/derivRight(xRight[0]+self.derivOffset)
            
            if self.debug:
                bR = xRight[0] - mR*yRight[0]
                x1R,y1R = (-10000, mR*-10000+bR)
                x2R,y2R = (10000, mR*10000+bR)
                plot((x1R,x2R),(y1R,y2R),'b')
                plot(yRight,xRight,'r.')
                plot(ysRight, xsRight, 'g')

            caR = -1*np.arctan(mR)*180/np.pi
            if caR < 0: caR = 180+caR 
            
        # Only generate plots in debug mode
        if makePlot and self.debug:
            axhline(self.h - self.bed) 
            fileName = self.workDir + count + "-plt.svg"
            
            if self.isVideo: # If video
                xlim(self.w/2-120,self.w/2+120) # 67-307
                ylim(self.h-self.bed-20,self.h-40) # 40-250
            else: # If tilt image
                xlim(self.w/2-500,self.w/2+500) # Wider
                ylim(self.h-self.bed-20,self.h+240) # Same
            title(fileName, size=8)
    
            savefig(fileName)#, dpi=self.dpi, papertype='b10', 
                    #orientation='portrait', bbox_inches='tight')
        
            clf() # Clear figure
        
        return caL,caR 


    def findBubbleEdge(self, arr, useBed=True):
        """ Find the top edge of a goniometer bubble.
        Input:  Transposed numpy array of filtered image.
        Output:  Returns false if no edge found, else edge position.
        """
        # j is position in arr, pxVal is value of the pixel
        # cannot used np.iternd here it only iterates forward
        for j,pxVal in enumerate(arr):
            if j > self.bed and useBed: #  If no value found before self.bed
                return None
            if pxVal < 30: # if a point is found return that points y value
                return j
        
        return None # If some how a horizontal white line exists in the image


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

            # Basically don't start looking until at least 120 px
            # This avoids noise in the top left corner.
            if i < self.syringMinPos:
                continue

            # Condition for finding left side of syringe
            if pxVal < 30 and syringeFound == False: 
    
                # Look at every point distance 3 from syringe to distance 100
                for k in xrange(i-self.sygOffset,i-self.bubMax,-2):
                    
                    # Basically break loop if you are getting near image edge
                    if k < 5: break
                    
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
    
                # Look at every point distance sygOffset to bubMax
                for k in xrange(i+self.sygOffset,i+self.bubMax,2):
                    
                    if k > self.w-5: break
                    
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
        
        # X is a function of Y, Y is a relation of X
        xFromLeft = [] # Actually a y value but becomes x viewed from left
        yFromLeft = [] # Actually an x value but becomes y viewed from left
        xFromRight = [] # Actually a y value but becomes x viewed from right
        yFromRight = [] # Actually an x value but becomes y viewed from right
        
        # Fit horizontally using an offset derived from xLeft yLeft
        # iterate over rows from BED up to highest bubble point
        if xLeft != [] and len(xLeft) > self.minFitPoints:
            for x in xrange(self.bed+1,(self.h-max(yLeft)),-1):  
                
                # Look for bubble edge horizontally
                # Ignore self.bed because you are looking horizontally
                y = self.findBubbleEdge(img[x][min(xLeft)-5:], useBed=False)
                
                if y == None: break # Don't add null data points
                y = y + min(xLeft)-5
                
                xFromLeft.append(self.h-x)
                yFromLeft.append(y)

        ### Major issue encountered here
        # For some reason you must do this assignment, you cannot do all 
        # these operations and [::-1] in the same line. Also np.nditer only 
        # iterates in one direction no matter what you send it... 
        # very confusing implementation.
        if xRight != [] and len(xRight) > self.minFitPoints:
            for x in xrange(self.bed+1,(self.h-max(yRight)),-1):
                
                # Look at row x from +5 px right of edge of bubble
                # Required separate line unsure why...
                array = img[x][:max(xRight)+5] 
                
                # Send reversed array because you need look from right
                y = self.findBubbleEdge(array[::-1], useBed=False)
                
                if y == None: break
                y = max(xRight)-y+5
                
                xFromRight.append(self.h-x)
                yFromRight.append(y)        

        # Estimate diameter of bubble
        if xLeft != [] and xRight != []:
            diameter = max(xRight) - min(xLeft)
        elif xLeft != []:
            # diameter with estimated syringe width
            diameter = max(xLeft) - min(xLeft) + self.sygOffset*2 + 4
        elif xRight != []:
            # diameter with estimated syringe width
            diameter = max(xRight) - min(xRight) + self.sygOffset*2 + 4
        else:
            diameter = None

        caL, caR = self.fitContactAngle(xFromLeft, yFromLeft, xFromRight, 
                                            yFromRight, count)

        # Return data or none if no data points found (diameter, caL, caR)   
        return diameter, caL, caR


    # mouse callback function
    def filterImg(self, img, rot_Matrx, imgNum, makePlot=True):
        """Filters and saves every Nth image in a video with getContactAngle()
        
        Input:
            self.itemPath = a string, path of video to be analyzed
            self.workDir = a string, working directory
            Parameters from config.py
        
        Output:
            A cropped, filtered, rotated .png (in debug mode)
            A call to find syringe edge
            A csv log file
        """
        
        
        if self.isVideo: # Basically if you're not processing a video
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
            
            # Video specific parameters for debug plotting
            savePath = self.workDir+imgNum+'.png'
            img_Nom = cv2.cvtColor(img_Nom,cv2.COLOR_BGR2GRAY)
            img_Debug = np.hstack((img,img_Nom))              
        else:
            # Filter
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.add(img,self.br)
            r,img = cv2.threshold(img,self.th,255,cv2.THRESH_BINARY)
            
            # Get width and height from black and white image
            self.h,self.w = img.shape
            
            # Mask filtered image to provide fake syringe
            mask = np.zeros([self.h,self.w], np.uint8)
            colMax = int(self.w/2 + 2)
            colMin =  int(self.w/2 -2)
            mask[0:5,colMin:colMax] = 255
            img = cv2.subtract(img,mask)

            # Image specific parameters for debug plotting
            savePath = self.itemPath[:-4]+'-fltrd.png'
            img_Debug = img
        
        if self.debug and makePlot: # Save debugging image 
   
            cv2.imwrite(savePath, img_Debug)
        
        return img



    def getAvgContactAngle(self, frames, angles):
        """ Get average advancing contact angle.  This algo assumes the best
        average can be found between frames 240-270
        Input: list of frames, list of angles
        Output: float value for average advancing contact angle
        """
        
        # frames = frames
        # Make sure array is not empty
        if len(frames) != len(angles):
            lmin = lmax = None
        if len(frames) > 280/self.n:
            lmin = int(240/self.n)
            lmax = int(270/self.n)
        elif len(frames) > 10:
            lmin = int(len(frames)/2-2)
            lmax = int(len(frames)/2+2)
        else:
            lmin = lmax = None
        
        # Calculate average
        if lmin:
            avgL = angles[lmin:lmax]
            avgAdvCA = reduce(lambda x, y: x + y, avgL)/len(avgL)
        else:
            avgAdvCA = None
        
        return avgAdvCA

    def measureVideo(self):
        """Filters and saves every Nth image in a video with getContactAngle()
        
        Input:
            self.itemPath = a string, path of video to be analyzed
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
        diameters = []
        diameterFrames = []
        makePlot = False
        
        # Start log file
        name = self.itemPath.split('\\')[-1:][0].split('_BED')[0]
        logPath = self.workDir + name + '_data.csv'
        logFile = open(logPath, 'wb')
        log = csv.writer(logFile, delimiter=',', escapechar='|',
                          quoting=csv.QUOTE_NONE)
        # Headers
        runParams = [
                     'bed='+str(self.bed),
                     'N='+str(self.n),
                     'r='+str(self.r),
                     'roi='+str(self.roi),
                     'mask='+str(self.mask),
                     'br='+str(self.br),
                     'th='+str(self.th),
                     'dpi='+str(self.dpi), 
                     'debug='+str(self.debug),
                     'resetbed='+str(self.resetbed),
                     'syringeMinPos='+str(self.syringMinPos),
                     'sygOffset='+str(self.sygOffset),
                     'bubMax='+str(self.bubMax),
                     'derivOffset='+str(self.derivOffset),
                     'polyDeg='+str(self.polyDeg),
                     'minFitPoints='+str(self.minFitPoints),
                     'w='+str(self.w),
                     'h='+str(self.h),
                     'assignBeds='+str(self.assignBeds), 
                      ]
  
        log.writerow(runParams)
        log.writerow(['frameNumber','diameter','leftCA','rightCA']) 
                
        # Rotation matrix centered in self.roi and rotated by self.r degrees
        x_cent = int((self.roi[0]+self.roi[1])/2)
        y_cent = int((self.roi[2]+self.roi[3])/2)
        rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),self.r,1)
        
        # Start reading input video
        cap = cv2.VideoCapture(self.itemPath)
            
        while True:
            ret, frame = cap.read()
            
            # If frameNum/N=0 is returned do analysis
            if ret and (count % self.n) == 0 and count >= self.vsf:          
                            
                imgNum = string.zfill(count,4) # Change 1 to 0001
                
                # Filter image
                img = self.filterImg(frame, rot_Matrx, imgNum)
                
                # Get contact angles and plot results 
                diameter, caL, caR = self.findSyringeEdge(img, img.T, imgNum)

                if self.debug:           
                    print imgNum, diameter, caL, caR
                
                # Append to lists for plots later
                if caL:
                    leftAngles.append(caL)
                    leftFrames.append(count)
                if caR:
                    rightAngles.append(caR)
                    rightFrames.append(count)
                if diameter:
                    diameters.append(diameter)
                    diameterFrames.append(count)
                
                # Output data to log file
                log.writerow([count, diameter, caL, caR])
                     
            if count > self.vef or not ret:
                cap.release()
                break
            
            count += 1

        logFile.close() # Close your log file

        # Plot 
        if diameters != []:
            makePlot = True
            plot(diameterFrames, diameters, 'g')
        
        if leftAngles != []:
            makePlot = True
            plot(leftFrames,leftAngles,'r')
            
        if rightAngles != []:
            makePlot = True
            plot(rightFrames,rightAngles,'b')
            
        if makePlot:
            ylabel('ContactAngle (Degrees), diameter (Px)', size=8)
            xlabel('Image Number', size=8)
            title('TEST001 Contact Angle & diameter vs Image Number', size=8)
            p1 = plt.Rectangle((0, 0), 1, 1, fc='g')
            p2 = plt.Rectangle((0, 0), 1, 1, fc='r')
            p3 = plt.Rectangle((0, 0), 1, 1, fc='b')
            legend([p1,p2,p3], ['diameter','leftCA','rightCA'], 
                   loc=1,prop={'size':8})
            
            plotPath = self.workDir + name + '_data.png'
            savefig(plotPath, dpi=300, papertype='b10', 
                    orientation='portrait', bbox_inches='tight')
            clf() # Clear your figure
        
        # Get average advancing contact angles
        avgAdvCALeft = self.getAvgContactAngle(leftFrames, leftAngles)
        avgAdvCARight = self.getAvgContactAngle(rightFrames, rightAngles)
        
        return max(diameters), avgAdvCALeft, avgAdvCARight
    
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
            k = cv2.waitKey(4) & 0xFF # Basically delay 1000 ms
            #print 'BED is currently: ', self.bed
            if k == 27: # Press escape to exit
                break
            
        cv2.destroyAllWindows()
        
    def getBED(self):
        """Attempts to pull BED from file name else prompts user to select it.
        
        Input: videoPath
        Output: renames video file to store BED
        """
        
        # Pull BED from file name assumes this file structure:
        # [Name of any format exclude the string _BED]_BED###.MOV
        # ### indicating an integer value for BED counted down from the top
        # of the cropped and filtered image.
        if '_BED' in self.itemPath and self.resetbed == False:
            self.bed = int(self.itemPath.split('BED')[1][:-4])
            return
        
        # Check to see what's being worked on then manually get BED, let's 
        # choose frame 250 for .MOVs
        if '.jpg' in self.itemPath or '.png' in self.itemPath:
            img = cv2.imread(self.itemPath)
            
            # Get width, height and channels
            self.h,self.w,channels = img.shape
            
            img = self.filterImg(img, None, None, makePlot=True)
            self.userSelectBed(img)
        else: # Process as video

            count = 0
            
            # Rotation matrix centered in self.roi and rotated by self.r degrees
            x_cent = int((self.roi[0]+self.roi[1])/2)
            y_cent = int((self.roi[2]+self.roi[3])/2)
            rot_Matrx = cv2.getRotationMatrix2D((x_cent,y_cent),self.r,1)
            
            # Start reading input video
            cap = cv2.VideoCapture(self.itemPath)
            
            # User selects BED from filtered image
            while True:
                ret, frame = cap.read()
                
                if count == 250: #Arbitrary mid point         
                    imgNum = '0250'
                    # Filter image and get BED
                    img = self.filterImg(frame, rot_Matrx, imgNum, 
                                         makePlot=False)
                    self.userSelectBed(img)
                    cap.release()
                    break
                
                if count > self.vef or not ret:
                    
                    cap.release()
                    break
                
                count += 1

        # Store self.bed in file name
        ext = self.itemPath.split('.')[-1]         
        # Remove _BED###.ext
        if self.resetbed and '_BED' in self.itemPath: 
            name = self.itemPath.split('_BED')
            bedName = name[0]+'_BED'+str(self.bed)+'.' + ext
        else: # Remove .ext
            bedName = self.itemPath[:-4]+'_BED'+str(self.bed)+'.' + ext

        if self.itemPath != bedName:
            print 'Renaming: ', self.itemPath
            print 'New Name: ', bedName
            os.rename(self.itemPath, bedName)            
            self.itemPath = bedName
        else:
            print 'Not renaming, BED unchanged.'
            print 'Name: ', self.itemPath
            print 'New Name: ', bedName
    
    def removeBED(self):
        """ Remove _BED from file names
        """
        
        # Rename file so this doesn't need to happen again
        if '_BED' in self.itemPath: # Remove _BED###.MOV
            name = self.itemPath.split('_BED')
            bedName = name[0]+'.MOV'
            print 'Renaming: ', self.itemPath
            print 'New Name: ', bedName
            

    
    def analyzeVideos(self):
        """ Executes getContactAngle on an individual video or recursively over 
        an entire folder tree. 
        
        Input:
        Parameters from config.py
        
        Output:
        In non debug mode it deletes folders with duplicate names.
        """
        
        if '.MOV' in self.path: # For individual file call directly
            self.workDir = self.path[:-4]+'\\'
            self.itemPath = self.path
            print 'Analyzing: ', self.itemPath
            
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
            
            # Set up a master log file, timestamp prevents overwriting
            mlPath = 'D:\\_Work\\0000-AFCC_MetalFSU_A2\\masterLog_'
            ts = str(time.time())[2:-3] # timestamp
            mlPath = mlPath + ts + '.csv'
            print 'Saving master data to: ', mlPath
            masterLogFile = open(mlPath, 'wb')
            mlog = csv.writer(masterLogFile, delimiter=',', 
                              quoting=csv.QUOTE_NONE)
            mlog.writerow(self.headers)
            
            for (root, subFolders, files) in os.walk(self.path):
                #Second condition forces this to run in only the index dir
                for item in files:
                    
                    if '.MOV' in item:
                        
                        if 'FIX' in item: continue # Ignore pre-filtered video                        
                        self.itemPath = root+'\\'+item

                        # Only remove _BED from file names
                        if self.removeBeds:
                            print 'Remove BEDs.'
                            self.removeBED()
                            continue
                        
                        # Only assign beds no analysis
                        if self.assignBeds: 
                            print 'Assigning BEDs.'
                            self.getBED() # Pull BED from filename or ask user
                            continue
                        else:
                            self.getBED()
                        
                        print 'Analyzing: ', self.itemPath
                        self.workDir = self.itemPath[:-4]+'\\' # Remove .MOV
                        
                        if os.path.isdir(self.workDir) and not self.debug:
                            print 'Deleting: ', self.workDir
                            shutil.rmtree(self.workDir)
                        
                        if not os.path.isdir(self.workDir):
                            print 'Making: ', self.workDir
                            os.mkdir(self.workDir)
                        
                        # Get averaged values for things we car about
                        d,caL,caR = self.measureVideo()
                        
                        # Splits 20140805-12_49_15-CA7_L002A_02_BED140.MOV
                        # to ['CA7', 'L002A', '02', 'BED140.MOV']
                        info = item.split('-')[2].split('_')
                        
                        # Log filename, folderID, fabID, sample#, region,
                        # run#, caL, caR, diameter, bed
                        row = [item, info[0], info[1][:-4], info[1][-4:-1],
                               info[1][-1], caL, caR, d, self.bed]
                        print row
                        
                        # Write data row to master log file
                        # Flush to prevent data loss in crash
                        mlog.writerow(row)
                        masterLogFile.flush() 
            
            masterLogFile.close()


    def analyzeImages(self):
        """Find contact angles from images
        
        Input: configured gCA class
        
        Output: calls many functions
        
        Note:  A lot of this reproduction is suggesting to me I need to a new 
        class at this point that inherits gCA... but I'm not going to get into 
        that now.
        
        """
        
        # Set up master log as usual
        mlPath = 'D:\\_Work\\0000-AFCC_MetalFSU_A2\\masterLog_Tilt_'
        ts = str(time.time())[2:-3] # Repeats at about 100 weeks
        mlPath = mlPath + ts + '.csv'
        print 'Saving master data to: ', mlPath
        masterLogFile = open(mlPath, 'wb')
        mlog = csv.writer(masterLogFile, delimiter=',', escapechar='|',
                          quoting=csv.QUOTE_NONE)

        runParams = [
                     'bed='+str(self.bed),
                     'br='+str(self.br),
                     'th='+str(self.th),
                     'dpi='+str(self.dpi), 
                     'debug='+str(self.debug),
                     'resetbed='+str(self.resetbed),
                     'syringeMinPos='+str(self.syringMinPos),
                     'sygOffset='+str(self.sygOffset),
                     'bubMax='+str(self.bubMax),
                     'derivOffset='+str(self.derivOffset),
                     'polyDeg='+str(self.polyDeg),
                     'minFitPoints='+str(self.minFitPoints),
                     'w='+str(self.w),
                     'h='+str(self.h),
                     'assignBeds='+str(self.assignBeds), 
                      ]
        
        mlog.writerow(runParams)
        
        mlog.writerow(self.headers)
        

        
        for (root, subFolders, files) in os.walk(self.path):
            #Second condition forces this to run in only the index dir
            for item in files:
                
                # Prevent oswalk from visiting subfolders:
#                 wRoot = 'D:\\_Work\\0000-AFCC_MetalFSU_A2\\20140810_TiltPhotos\\'
#                 if root != wRoot:
#                     continue
                if '-fltrd' in item: # ignore filtered images
                    continue
                
                
                if '.jpg' in item or '.png' in item:
                    
                    self.itemPath = root+'\\'+item
                    self.workDir = root+'\\'
    
                    # Only remove _BED from file names
                    if self.removeBeds:
                        print 'Remove BEDs.'
                        self.removeBED()
                        continue
                    
                    # Only assign beds no analysis
                    if self.assignBeds: 
                        print 'Assigning BEDs.'
                        self.getBED() # Pull BED from filename or ask user
                        continue
                    else:
                        self.getBED()
                    
                    print 'Analyzing: ', self.itemPath

                    # Load and filter images
                    img = cv2.imread(self.itemPath)
                    img = self.filterImg(img,None,None)
                    

                    
                    # Use algo to measure CA's
                    name = item[:-4]
                    d, caL, caR = self.findSyringeEdge(img, img.T, name)
                    print d, caL, caR # caL = receding, caR = advancing
                     
                    mat = item.split('_')[4] # Material
                    tiltAngle = int(item.split('Rotate')[1].split('_')[0])
                    idx = item.split('_')[5]
                    row = [item, mat, idx, tiltAngle, caL, caR, d, self.bed]
                    
                    # Write data row to master log file
                    # Flush to prevent data loss in crash
                    mlog.writerow(row)
                    masterLogFile.flush() 
        
        masterLogFile.close()
    

if __name__ == '__main__':
#     gCAVid = getContactAngle(varsVid, headersVid)
#     gCAVid.analyzeVideos()
    
    gCATilt = getContactAngle(varsImg, headersImg, isVideo=False)
    gCATilt.analyzeImages()

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
