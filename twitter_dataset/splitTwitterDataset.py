import os
import sys
import random
import time
import numpy as np
import CNN_tools



GROUND_TRUTH_PATH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/" # the r at the beginning solves some problems with the spaces in the path
IMAGES_PATH = r"/imatge/vcampos/work/twitter_dataset/images/resized/" # the r at the beginning solves some problems with the spaces in the path
instanceList = []
testFraction = 0.3
ground_truth_file = ""
prefix = ""


# First argument: 3/4/5 agrees; second argument: test split fraction
if (len(sys.argv) >= 3):
    try:
        testFraction = float(sys.argv[2])
        num = int(sys.argv[1])
        if (num == 3):
            ground_truth_file = "twitter_three_agrees.txt"
            prefix = "three_agrees"
        elif (num == 4):
            ground_truth_file = "twitter_four_agrees.txt"
            prefix = "four_agrees"
        elif (num == 5):
            ground_truth_file = "twitter_five_agrees.txt"
            prefix = "five_agrees"
        else:
            sys.exit('Wrong parameters')
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
    # Add the line to the list
    instanceList.append(line)
print ("Reading from the report: %.2f seconds" %(time.time()-t0))

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
# Write each subset to a new file, called "./prefix/train.txt" and "./prefix/test.txt" containing absolute path to the images and their labels
dir = GROUND_TRUTH_PATH+prefix+"_"+str(int(100-100*testFraction))+"-"+str(int(100*testFraction))+"/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in trainData:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in testData:
    file_test.write(IMAGES_PATH+line)
file_test.close()
print ("Writing subset files: %.2f seconds" %(time.time()-t0))

