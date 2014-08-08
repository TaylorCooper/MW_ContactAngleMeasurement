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