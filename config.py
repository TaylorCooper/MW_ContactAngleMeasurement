"""
    Author=
        Taylor Cooper
    Description=
        Load inputs for main=
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
                        
    Date Created=
        Aug 4, 2014 7=07=07 PM
    
    Arguments and Inputs=
        None
    Outputs=
        None
    Dependencies=
        None
                  
    History=                  
    --------------------------------------------------------------
    Date=    
    Author=    Taylor Cooper
    Modification=    
     --------------------------------------------------------------
"""
            
params = dict(
    # Always provide the trailing \\ for 'PATH' if it's a file path
    PATH = "D=\\GitHub\\workspace\\A2_work\\SampleAnalysis\\",
    VSF = 80,
    VEF = 660,
    N = 10,
    R = 2,
    ROI = (850,1080,800,1175), # row_min, row_max, col_min, col_max
    MASK = (90,280,255), # col_min, col_max, value outside mask
    BR = 90,
    TH = 160,
    DEBUG = True,
    RESETBED = False
)

