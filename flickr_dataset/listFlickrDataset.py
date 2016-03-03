import os
import sys
import random
import time
import numpy as np


'''IMAGES_PATH = r"/imatge/vcampos/projects/faces/affective/sentibank/Images_with_CC/bi_concepts1553/"
REPORT_PATH = r"/imatge/vcampos/work/flickr/" # the r at the beginning solves some problems with the spaces in the path
SAVE_FILE = r"/imatge/vcampos/work/flickr/ground_truth.txt"
'''
IMAGES_PATH = r"/imatge/vcampos/work/flickr/CCimages_256x256/"
REPORT_PATH = r"/imatge/vcampos/work/flickr/" # the r at the beginning solves some problems with the spaces in the path
SAVE_FILE = r"/imatge/vcampos/work/flickr/ground_truth_CC_256x256.txt"
instanceList = []


# Open the file
file = open(REPORT_PATH+'3244ANPs.txt', "r")


# Read the "header"/blank lines which do not contain information about the images
for i in range (0,3):
    file.readline()


# Loop through the document to obtain the list of ANPs+sentiment
while(True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Do not add the adjective "summary" lines
    elif (not(line[0]>='1' and line[0]<='9')):
        splitted_line = line.split()
        anp = splitted_line[0]
        sentiment = splitted_line[2][:-1] # removes the ']' from the end
        instanceList.append(anp + " " + sentiment) # Add the instance to the list
print "Total ANPs: " + str(len(instanceList))
file.close()



# Create the file where the list will be saved
dst_file = open(SAVE_FILE,"w")

t0 = time.time()
# Loop through the ANPs, list all the files in each subfolder and add them to the document
print "Listing files..."
imgExts = ["png", "bmp", "jpg"]
for anp in instanceList:
    anp_name = anp.split()[0]
    sentiment = anp.split()[1]
    anp_path = IMAGES_PATH+anp_name+"/"
    for path, dirs, files in os.walk(anp_path):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            dst_file.write(anp_path+fileName+" "+sentiment+"\n")
dst_file.close()
print ("Listing images in dest file: %.2f seconds" %(time.time()-t0))
