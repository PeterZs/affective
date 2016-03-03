import os, sys, time
import numpy as np
import cv2
import CNN_tools

PATH_TO_DATASET = ""
PATH_TO_RESULTS = "" # where the feature vectors will be saved

color = ('b','g','r') # OpenCV loads images in BGR format

if (len(sys.argv)>=3):
    try:
        PATH_TO_DATASET = str(sys.argv[1])
        PATH_TO_RESULTS = str(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit('Not enough arguments')


# Get the images in the path
listImages = os.listdir(PATH_TO_DATASET)

# Create the directory where the results will be saved if it does not exist already
CNN_tools.createDir(PATH_TO_RESULTS)

# Compute the descriptors and save them
print 'Computing color histograms...'
counter = 0
t0 = time.time()
for image in listImages:
    image_split = image.split('.',2)
    image_name = image_split[0]
    if (len(image_split)>2): # for those images named 1234.5.jpg
        image_name = image_split[0] + '.' + image_split[1]
    if(len(image_split[0])>0):
        t1 = time.time()
        feat = np.zeros((0,1))
        img = cv2.imread(PATH_TO_DATASET+image, 1) # load image in color
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            feat = np.concatenate((feat,hist), axis=0)
        del img
        np.save(PATH_TO_RESULTS+image_name,feat)
        counter+=1
        if(counter%10==0):
            print 'Progress: ' + str(counter) + '/' + str(len(listImages))


print 'Done. Total time spent = ' + str((time.time()-t0)/60) + 'min'
print 'Extracted ' + str(counter) + ' histograms'