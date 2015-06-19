import os, sys, time
import numpy as np
from sklearn.svm import SVC

TRAIN_GROUND_TRUTH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/test1/train.txt"
TEST_GROUND_TRUTH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/test1/test.txt"
FEATURES_PATH = r"/imatge/vcampos/work/handcrafted_features/twitter/colorHist/"

data_train = np.zeros((3*256,0))
data_test = np.zeros((3*256,0))
labels_train = np.zeros((0,1))
labels_test = np.zeros((0,1))

# Open files
file_train = open(TRAIN_GROUND_TRUTH, "r")
file_test = open(TEST_GROUND_TRUTH, "r")


# Load train data and ground truth
t0 = time.time()
while(True):
    line = file_train.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    values = line.split()
    labels_train = np.append(labels_train, int(values[1]))
    imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
    feat = np.load(FEATURES_PATH+imageName+".npy")
    data_train = np.append(data_train, feat, axis=1)
print ("Loading train data: %.2f seconds" %(time.time()-t0))
sys.stdout.flush()

# Load test data and ground truth
t0 = time.time()
while(True):
    line = file_test.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    values = line.split()
    labels_test = np.append(labels_test, int(values[1]))
    imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
    feat = np.load(FEATURES_PATH+imageName+".npy")
    data_test = np.append(data_test, feat, axis=1)
print ("Loading test data: %.2f seconds" %(time.time()-t0))
sys.stdout.flush()

# Close files
file_train.close()
file_test.close()

# Swap axes so data.shape[0]=labels.shape[0]
data_train = np.swapaxes(data_train, 0, 1)
data_test = np.swapaxes(data_test, 0, 1)

# Check dimensions
print 'Train labels: ', labels_train.shape
print 'Train data: ', data_train.shape
print 'Test labels: ', labels_test.shape
print 'Test data: ', data_test.shape
sys.stdout.flush()

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



'''
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

# Train SVM using train data
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Test SVM by predicting over test data
print(clf.predict([[-0.8, -1]]))
'''