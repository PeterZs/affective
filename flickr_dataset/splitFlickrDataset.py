import os
import sys
import random
import time
import numpy as np
import CNN_tools



GROUND_TRUTH_PATH = r"/imatge/vcampos/work/flickr/" # the r at the beginning solves some problems with the spaces in the path
instanceList = []
testFraction = 0.3
ground_truth_file = "ground_truth_CC_256x256.txt"
prefix = ""
sentiment_threshold = 0.5


# Choose the test fraction and the minimum sentiment value
if (len(sys.argv) >= 3):
    try:
        testFraction = float(sys.argv[1])
        sentiment_threshold = float(sys.argv[2])
    except:
        sys.exit('Wrong parameters')
else:
    sys.exit('Not enough parameters')


# Open the file
file = open(GROUND_TRUTH_PATH+ground_truth_file, "r")

t0 = time.time()
# Loop through the document
while(True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    values = line.split()
    sentiment = float(values[1])
    # Add the line to the list. Labels are positive (1) and negative (0)
    if abs(sentiment) >= sentiment_threshold:
        if sentiment > 0:
            sentiment = 1
        else:
            sentiment = 0
        instanceList.append(values[0] + " " + str(sentiment)+"\n")
print ("Reading from the report: %.2f miliseconds" %(1000.*(time.time()-t0)))

# Close the file
file.close()

# Shuffle the images in the list
random.shuffle(instanceList)


# Compute the amount of images that will be used for training/testing (1194 images in the dataset)
numTestImages = int(testFraction*len(instanceList))
numTrainImages = len(instanceList)-numTestImages


# Split the data into training/testing sets
trainData = instanceList[:-numTestImages]
testData = instanceList[-numTestImages:]

t0 = time.time()
# Write each subset to a new file
dir = GROUND_TRUTH_PATH+str(int(100-100*testFraction))+"-"+str(int(100*testFraction))+"/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in trainData:
    file_train.write(line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in testData:
    file_test.write(line)
file_test.close()
print ("Writing subset files: %.2f miliseconds" %(1000.*(time.time()-t0)))

print "Training images: " + str(numTrainImages)
print "Testing images: " + str(numTestImages)

