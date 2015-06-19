import numpy as np
import pylab as Plot
import tsne

# Paths
GROUND_TRUTH_PATH = "/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/test1/test.txt"
FEATURES_PATH = "/imatge/vcampos/work/twitter_dataset/feature_maps/test1/fc6/"

# Open file
file = open(GROUND_TRUTH_PATH, "r")

# Store images in a list
print "Formatting data..."
X = []
labels = []
while (True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line) == 0):
        break
    # Store the feature map in X and the label in labels
    [image_path,label] = line.split()
    labels.append(label)
    file_name=image_path.split("/")[-1].split(".")[0]
    X.append(np.load(FEATURES_PATH+file_name+".npy").squeeze(-1).squeeze(-1))
X = np.array(X,dtype='float64')
labels = np.array(labels,dtype='float64')
print "Done"

# Check shapes
print "\nX.shape: " + str(X.shape)
print "labels.shape: " + str(labels.shape)


# Compute t-sne
print "\nComputing t-SNE..."
Y = tsne.tsne(X, 2, 50, 5.0);
np.save("/imatge/vcampos/work/tsne_Y.npy",Y)
Plot.scatter(Y[:,0], Y[:,1], 20, labels);

