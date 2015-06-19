import numpy as np
import matplotlib.pyplot as plt
import sys


# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


# Show an image stored as a numpy array
def show_image(im_file):
    image = np.load(im_file)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    file = ""

    if (len(sys.argv) >= 2):
        try:
            file = str(sys.argv[1])
        except:
            sys.exit('Wrong parameters')
    else:
        sys.exit('Not enough parameters')


    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Load the data and plot it
    feat = np.load(file)
    plt.plot(feat.flat)
    plt.show()