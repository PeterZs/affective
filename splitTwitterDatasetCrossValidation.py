import os
import sys
import random
import time
import numpy as np
import CNN_tools


'''
 Splits Twitter "five agrees" dataset for 5-fold cross-validation
 Subsets: 5*176 = 880 images (the full set has 882 images, but this way every subset has the same size)
'''


GROUND_TRUTH_PATH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/" # the r at the beginning solves some problems with the spaces in the path
IMAGES_PATH = r"/imatge/vcampos/work/twitter_dataset/images/resized/" # the r at the beginning solves some problems with the spaces in the path
instanceList = []
fraction = 0.2
ground_truth_file = "twitter_five_agrees.txt"
subsetSize = 176


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
print ("Reading from the report: %.2f miliseconds" %(1000.*(time.time()-t0)))

# Close the file
file.close()

# Shuffle the images in the list
random.shuffle(instanceList)


# Compute the amount of images that will be used for training/testing (1194 images in the dataset)



# Split the data into training/testing sets
set1 = instanceList[:subsetSize]
set2 = instanceList[subsetSize:2*subsetSize]
set3 = instanceList[2*subsetSize:3*subsetSize]
set4 = instanceList[3*subsetSize:4*subsetSize]
set5 = instanceList[4*subsetSize:5*subsetSize]


t0 = time.time()
# Write the training/testing files for each experiment in a different folder
CNN_tools.createDir(GROUND_TRUTH_PATH+"5-fold_cross-validation/")

# Using subset 1 for testing
dir = GROUND_TRUTH_PATH+"5-fold_cross-validation/"+"test1/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set2+set3+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set1:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 2 for testing
dir = GROUND_TRUTH_PATH+"5-fold_cross-validation/"+"test2/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set3+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set2:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 3 for testing
dir = GROUND_TRUTH_PATH+"5-fold_cross-validation/"+"test3/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set3:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 4 for testing
dir = GROUND_TRUTH_PATH+"5-fold_cross-validation/"+"test4/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set3+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set4:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 5 for testing
dir = GROUND_TRUTH_PATH+"5-fold_cross-validation/"+"test5/"
CNN_tools.createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set3+set4:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set5:
    file_test.write(IMAGES_PATH+line)
file_test.close()

print ("Writing subset files: %.2f miliseconds" %(1000.*(time.time()-t0)))
