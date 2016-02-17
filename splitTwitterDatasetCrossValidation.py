import os
import sys
import random
import time
import numpy as np
import argparse


'''
 Splits Twitter "X agrees" dataset for 5-fold cross-validation
 Subsets:
    five-agree: 5*176 = 880 images (the full set has 882 images, but this way every subset has the same size)
    four-agree: 5*223 = 1115 images (the full set has 1116 images, but this way every subset has the same size)
    three-agree: 5*253 = 1265 images (the full set has 1269 images, but this way every subset has the same size)
'''


GROUND_TRUTH_PATH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/" # the r at the beginning solves some problems with the spaces in the path
IMAGES_PATH = r"/imatge/vcampos/work/twitter_dataset/images/resized/" # the r at the beginning solves some problems with the spaces in the path
instanceList = []
fraction = 0.2
ground_truth_file = "twitter_four_agrees.txt"
output_folder = "5-fold_cross-validation_four_agrees/"
subsetSize = 223


def createDir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print 'Create dir: ', directory
    return directory



parser = argparse.ArgumentParser(description='Splits Twitter x-agree dataset for 5-fold cross-validation')
parser.add_argument('subset', help='Subset to split (3, 4 or 5)')
args = parser.parse_args()

agrees = int(args.subset)

if agrees == 3:
    ground_truth_file = "twitter_three_agrees.txt"
    output_folder = "5-fold_cross-validation_three_agrees/"
    subsetSize = 253
elif agrees == 4:
    ground_truth_file = "twitter_four_agrees.txt"
    output_folder = "5-fold_cross-validation_four_agrees/"
    subsetSize = 223
elif agrees == 5:
    ground_truth_file = "twitter_five_agrees.txt"
    output_folder = "5-fold_cross-validation_five_agrees/"
    subsetSize = 176
else:
    sys.exit("Wrong parameters. Chosse between 3, 4 and 5 - agrees subset")


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
createDir(GROUND_TRUTH_PATH+output_folder)

# Using subset 1 for testing
dir = GROUND_TRUTH_PATH+output_folder+"test1/"
createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set2+set3+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set1:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 2 for testing
dir = GROUND_TRUTH_PATH+output_folder+"test2/"
createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set3+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set2:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 3 for testing
dir = GROUND_TRUTH_PATH+output_folder+"test3/"
createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set4+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set3:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 4 for testing
dir = GROUND_TRUTH_PATH+output_folder+"test4/"
createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set3+set5:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set4:
    file_test.write(IMAGES_PATH+line)
file_test.close()

# Using subset 5 for testing
dir = GROUND_TRUTH_PATH+output_folder+"test5/"
createDir(dir)
file_train = open(dir+"train.txt", "w")
for line in set1+set2+set3+set4:
    file_train.write(IMAGES_PATH+line)
file_train.close()

file_test = open(dir+"test.txt", "w")
for line in set5:
    file_test.write(IMAGES_PATH+line)
file_test.close()

print ("Writing subset files: %.2f miliseconds" %(1000.*(time.time()-t0)))
