import sys
import os
import time
import numpy as np
import CNN_tools

PATH_TO_DATASET = ""
PATH_TO_RESULTS = "" # where the feature vectors will be saved

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

# Load CNN from ImageNet
net = CNN_tools.loadImageNetCNN()

# Create the directory where the results will be saved if it does not exist already
CNN_tools.createDir(PATH_TO_RESULTS)

# Compute the descriptor and save it
t0 = time.time()
for image in listImages:
    image_split = image.split('.',2)
    image_name = image_split[0]
    if (len(image_split)>2): # for those images named 1234.5.jpg
        image_name = image_split[0] + '.' + image_split[1]
    if(len(image_split[0])>0):
        t1 = time.time()
        prediction = CNN_tools.predict(net, PATH_TO_DATASET+image)
        print 'Time spent = ' + str(time.time()-t1) + 's. Saving prediction vector for ' + str(image) + '...'
        np.save(PATH_TO_RESULTS+image_name,prediction)


print 'Done. Total time spent = ' + str((time.time()-t0)/60) + 'min'