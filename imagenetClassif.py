import numpy as np
import matplotlib.pyplot as plt
import time

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/opt/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/imatge/vcampos/caffe/models/bvlc_reference_caffenet/new/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = '/imatge/vcampos/caffe/examples/images/cat.jpg'
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
imagenet_labels_filename = '/imatge/vcampos/caffe/data/ilsvrc12/synset_words.txt'

if (len(sys.argv)>=2):
    try:
        IMAGE_FILE = str(sys.argv[1])
    except:
        sys.exit('The given arguments are not correct')
else:
    print 'Using ' + IMAGE_FILE


net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_phase_test()
net.set_mode_gpu()
net.set_mean('data', MEAN_FILE)  # ImageNet mean
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.set_input_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]


input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

t0 = time.time()
#prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image], oversample=True)


print 'Prediction time = ' + str(time.time()-t0) + 's'
print 'prediction shape:', prediction[0].shape
print 'predicted class:', prediction[0].argmax()
print 'score', prediction[0].item(prediction[0].argmax())


try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
    print 'Labels: '
    print labels[top_k]
except:
    print 'Error loading synset_words.txt'