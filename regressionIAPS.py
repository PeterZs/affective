import os
import sys
import random
import time
import numpy as np
from sklearn import datasets, linear_model


class IAPSInstance:
    def __init__(self, id, valmn, valsd, aromn, arosd):
        self.id = id
        self.valmn = float(valmn)
        self.valsd = float(valsd)
        self.aromn = float(aromn)
        self.arosd = float(arosd)

    def toString(self):
        return 'id='+str(self.id)+', valmn='+str(self.valmn)+', valsd='+str(self.valsd)+', aromn='+str(self.aromn)+', arosd='+str(self.arosd)



data_type = 1 # 0 for predictions, 1 for features

REPORT_PATH = r"/imatge/vcampos/projects/faces/affective/iaps/IAPS 2008 1-20/IAPS Tech Report/" # the r at the beginning solves some problems with the spaces in the path
instanceList = []
testPercentage = 0.3

if(data_type==0): # changes path between iaps_predictions and iaps_features
    DATA_PATH = "/imatge/vcampos/work/iaps_predictions/"
else:
    DATA_PATH = "/imatge/vcampos/work/iaps_features/"


# Open the file
file = open(REPORT_PATH+'AllSubjects_1-20.txt', "r")


# Read the "header" lines which do not contain information about the images
for i in range (0,7):
    file.readline()

t0 = time.time()
# Loop through the document
while(True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Obtain the desired values
    values = line.split()
    aux = IAPSInstance(values[1], values[2], values[3], values[4], values[5])
    # Add the instance to the list
    instanceList.append(aux)
print ("Reading from the report: %.2f seconds" %(time.time()-t0))

# Shuffle the images in the list
random.shuffle(instanceList)


# Compute the amount of images that will be used for training/testing (1194 images in the dataset)
numTestImages = int(testPercentage*len(instanceList))
numTrainImages = len(instanceList)-numTestImages


# Split the data into training/testing sets
testData = instanceList[-numTestImages:]
trainData = instanceList[:-numTestImages]

t0 = time.time()
# Obtain training/testing values for arousal mean and valence mean
train_aromn = []
test_aromn = []
train_valmn = []
test_valmn = []
for inst in trainData:
    train_aromn.append(inst.aromn)
    train_valmn.append(inst.valmn)
for inst in testData:
    test_aromn.append(inst.aromn)
    test_valmn.append(inst.valmn)


# Load and prepare the features
train_feat = []
test_feat = []
for inst in trainData:
    id = inst.id
    if (data_type==0):
        feat = np.load(DATA_PATH+id+'.npy')[0]
    else:
        feat = np.load(DATA_PATH+id+'.npy')
        feat = feat.squeeze(1)
        feat = feat.swapaxes(0,1)
        feat = feat[0]
    train_feat.append(feat)
for inst in testData:
    id = inst.id
    if (data_type==0):
        feat = np.load(DATA_PATH+id+'.npy')[0]
    else:
        feat = np.load(DATA_PATH+id+'.npy')
        feat = feat.squeeze(1)
        feat = feat.swapaxes(0,1)
        feat = feat[0]
    test_feat.append(feat)

print ("Loading and preparing features and results: %.2f seconds" %(time.time()-t0))

# Create linear regression objects
regr_aromn = linear_model.LinearRegression()

# Train the model using the training sets
t0 = time.time()
regr_aromn.fit(train_feat, train_aromn)
print ("Train the model: %.2f seconds" %(time.time()-t0))


# Predict over the test data
predictions_aromn = regr_aromn.predict(test_feat)


i = 0
for inst in testData:
    print "Theoretical aromn: " + str(inst.aromn)
    print "Predicted aromn: " + str(predictions_aromn[i])
    print "---------"
    i+=1


# The mean square error
print("Residual sum of squares (aromn): %.2f"
      % np.mean((predictions_aromn - test_aromn) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr_aromn.score(test_feat, test_aromn))

