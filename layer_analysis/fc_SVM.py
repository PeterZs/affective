import os, sys, time
import numpy as np
from sklearn.svm import SVC
import subprocess

TRAIN_GROUND_TRUTH = r"/imatge/vcampos/work/flickr/80-20/train.txt"
TEST_GROUND_TRUTH = r"/imatge/vcampos/work/flickr/80-20/test.txt"
FEATURES_PATH = r"/imatge/vcampos/work/flickr/feature_maps/"
LAYER_NAME = "fc7"

if (len(sys.argv)>=2):
    try:
        LAYER_NAME = str(sys.argv[1])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit('Not enough arguments. Layer name expected.')

FEATURES_PATH = FEATURES_PATH + LAYER_NAME + '/'

data_train = []
data_test = []
labels_train = []
labels_test = []

# Get the amount of lines in each file
num_train_images = int( (subprocess.Popen( 'wc -l {0}'.format( TRAIN_GROUND_TRUTH ), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0] )
num_test_images = int( (subprocess.Popen( 'wc -l {0}'.format( TEST_GROUND_TRUTH ), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0] )

# Open files
file_train = open(TRAIN_GROUND_TRUTH, "r")
file_test = open(TEST_GROUND_TRUTH, "r")

print 'Using feature maps from layer ' + LAYER_NAME
print 'Loading feature maps...'
sys.stdout.flush()

# Load train data and ground truth
t0 = time.time()
counter = 0
while(True):
    line = file_train.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    values = line.split()
    labels_train.append(int(values[1]))
    imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
    feat = np.load(FEATURES_PATH+imageName+".npy")
    data_train.append(feat.flatten())
    # Update the counter and show progress every 10 iterations
    counter += 1
    if (counter%10 == 0):
        print 'TRAIN: Loaded ' + str(counter) + '/' + str(num_train_images) + ' feature maps'
        sys.stdout.flush() # without flushing, no prints are shown until the end
print ("Loading train data: %.2f hours" %((time.time()-t0)/3600))
sys.stdout.flush()

# Load test data and ground truth
t0 = time.time()
counter = 0
while(True):
    line = file_test.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    values = line.split()
    labels_test.append(int(values[1]))
    imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
    feat = np.load(FEATURES_PATH+imageName+".npy")
    data_test.append(feat.flatten())
    # Update the counter and show progress every 10 iterations
    counter += 1
    if (counter%10 == 0):
        print 'TEST: Loaded ' + str(counter) + '/' + str(num_test_images)  + ' feature maps'
        sys.stdout.flush() # without flushing, no prints are shown until the end
print ("Loading test data: %.2f hours" %((time.time()-t0)/3600))
sys.stdout.flush()

# Close files
file_train.close()
file_test.close()

# Train SVM using train data
clf = SVC(kernel='linear', C=1.0)
print 'Training SVM...'
sys.stdout.flush()
t0 = time.time()
clf.fit(data_train, labels_train)
print ("Training SVM: %.2f seconds" %(time.time()-t0))
sys.stdout.flush()


# Test SVM
accuracy = clf.score(data_test, labels_test)
print ("Accuracy = %.2f%%" %(100.*accuracy))
sys.stdout.flush()