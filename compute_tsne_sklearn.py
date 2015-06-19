import numpy as np
import scipy.io
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold


SCATTER_PLOT = True  # toggles the visualization (deactivate when using srun)


# Next line to silence pyflakes. This import is needed.
Axes3D


if (len(sys.argv)>=3):
    try:
        layer = str(sys.argv[1])
        subset = str(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit("Not enough arguments. Run 'python compute_tsne_sklearn layer subset'")


# Paths
GROUND_TRUTH_PATH = "/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/"+subset+"/test.txt"
FEATURES_PATH = "/imatge/vcampos/work/twitter_dataset/feature_maps/"+subset+"/"+layer+"/"

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

# Set color for each label
neg_col = 'red'
pos_col = 'green'
color = []
for i in range(0, len(labels)):
    if (labels[i] == 1.0):
        color.append(pos_col)
    else:
        color.append(neg_col)
color = np.array(color)

# Compute t-sne
print "\nComputing t-SNE..."
tsne = manifold.TSNE(n_components=2, random_state=0, init='pca')
Y = tsne.fit_transform(X);
print "Done"
print "Saving results..."
scipy.io.savemat('/imatge/vcampos/work/pyxel/tools/affective/karpathy_tsne/val_embed_'+subset+'-'+layer+'.mat', mdict={'Y': Y})
print "Done"

if SCATTER_PLOT:
    print "Creating scatter plot..."
    plt.scatter(Y[:,0], Y[:,1], 10, c=color, cmap=plt.cm.Spectral);
    plt.title("t-SNE (Layer:"+layer+", Subset:"+subset+")")
    plt.axis('tight')
    plt.show()

