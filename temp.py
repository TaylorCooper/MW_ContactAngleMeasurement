        # Open output video    
        outName = outPath + 'Filtered.avi'
        
        # Output Filename, compression, fps, (frame_W, frame_H)
        # See notes below for possible compression formats
        video = cv2.VideoWriter(outName, cv2.cv.CV_FOURCC('I','4','2','0')
                                ,25,(self.w,self.h))
								
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
			
		# Possible compression formats: M J P G, P I M 1, I 4 2 0
        # Translation: Mjpg, i dont know, uncompressed avi
        video = cv2.VideoWriter(outName, cv2.cv.CV_FOURCC('P','I','M','1'),
                                25,(self.w,self.h))
								
								
	# Plot right  
    # Note the code below is confusing because I'm plotting from then rotating by 90
    # Convert to np.arrays
    xFromRight = np.asarray(xFromRight)
    yRight = np.asarray(yFromRight)

    # Fit Right side data set with 3rd degree polynomial
    coefRight = np.polyfit(xFromRight,yFromRight,10)
    polyRight = np.poly1d(coefRight)

    # Get values to plot fit, assume min and max locations
    xsRight = arange(xFromRight[0], xFromRight[-1], 0.1)
    ysRight = polyRight(xsRight)
    
    # Get derivate @ BED and values to plot dervative
    derivRight = polyRight.deriv()
    mL = 1/derivRight(xFromRight[0]) # Invert because flipping graph
    bL = xFromRight[0] - mL*yFromRight[0] # Also inverted
    x1L,y1L = (-1000, mL*-1000+bL)
    x2L,y2L = (1000, mL*1000+bL)
    
    # Plot Right lines
    plot((x1L,x2L),(y1L,y2L),'b')
    plot(yFromRight,xFromRight,'r.') # Inverted
    plot(ysRight, xsRight, 'g') # Inverted
    
    print 'CA: ', 180-np.absolute((np.arctan(mL)*180/np.pi))