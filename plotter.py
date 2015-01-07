"""
    Author:
        Taylor Cooper
    Description:
        Purpose goes here.                
    Date Created:
        Aug 29, 2014 1:53:50 PM
    
    Arguments and Inputs:
        None
    Outputs:
        None
    Dependencies:
        None
                  
    History:                  
    --------------------------------------------------------------
    Date:    
    Author:    Taylor Cooper
    Modification:    
     --------------------------------------------------------------
"""

# Basic plotting imports
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import pandas as pd  # Pandas dataframes very cool for data analysis stuff
import numpy as np  # Numpy is math functionality 
import os  # Basic OS functions

class caPlotter():
    """
    Description:
    Input:
    Output:
    """
    
    def __init__(self, var):
        self.var = var

    def myFunc(self):
        """
        Description:
        Input:
        Output:
        """

dataPath = "D:\\_Work\\0000-AFCC_MetalFSU_A2\\2014.08.28_Analysis\\"

if __name__ == '__main__':
    
    
    
    # For loop that looks for all files ending in .txt in the file path above
    for root, subFolders, files in os.walk(dataPath):
        
        for aFile in files:
            
            if aFile[-4:] == '.txt':
                path = dataPath + aFile
                print "Analyzing: " + path
     
                df = None
                df = pd.read_csv(path, sep=',', skiprows=5, header=None)  # Skip first 5 rows, no head create matrix with data
                
                # Reference the columns of your matrices
                x = df[0]
                y1 = df[1]
                y2 = df[2]
                
                # Plot left CA and right CA vs image number and add a legend and axis labels
                plot(x,y1,'r')
                plot(x,y2,'b')
                ylabel('ContactAngle (Degrees)', size=8)
                xlabel('Image Number', size=8)
                title('TEST001 Contact Angle vs Image Number', size=8)
                p1 = plt.Rectangle((0, 0), 1, 1, fc='r')
                p2 = plt.Rectangle((0, 0), 1, 1, fc='b')
                legend([p1,p2], ['Left CA','Right CA'], loc=1,prop={'size':8})
                
                # Save the output as a png
                savefig(path[:-4], dpi=350, papertype='b10', orientation='portrait', bbox_inches='tight')#, format='pdf')
                show()
     