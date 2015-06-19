import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/opt/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/imatge/vcampos/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'


def createDir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print 'Create dir: ', directory
    return directory


# Load any pretrained net
def loadNetCNN(model_file, pretrained, mean_file, dims, ch_swap, input_scale):
    net = caffe.Classifier( model_file, pretrained, gpu=True, image_dims=dims)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data', mean_file) # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.set_channel_swap('data', ch_swap)
    net.set_input_scale('data', input_scale)
    return net


# Load ImageNet CNN
def loadImageNetCNN():
    net = loadNetCNN('/imatge/vcampos/caffe/models/bvlc_reference_caffenet/new/deploy.prototxt',
                      '/imatge/vcampos/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                      '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.set_input_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    return net


# Resize an image so its shortest dimension matches newDim
def resizeImageShortestDim(im, newDim):
    if(im.shape[0]<im.shape[1]): # height<width
        aspectRatio = 1.*im.shape[0]/im.shape[1]
        return caffe.io.resize_image(im, (int(newDim*aspectRatio),newDim))
    else: # height>width
        aspectRatio = 1.*im.shape[1]/im.shape[0]
        return caffe.io.resize_image(im, (newDim,int(newDim*aspectRatio)))


def computeCNNFeatures(net, image_file, save_flag=False, output_dir='./features/', layer='fc7'):
    if(save_flag): # Create the directory if it does not exist
        createDir(output_dir)
    # Load image
    input_image = caffe.io.load_image(image_file)
    #Resize the image
    #input_image = resizeImageShortestDim(input_image, 256)
    prediction = net.predict([input_image], oversample=False)
    #print 'Image dimensions: ' + str(net.deprocess('data', net.blobs['data'].data[0]).shape)
    if(save_flag): # Save the features (and the image) as an array
        np.save(output_dir + layer +'_features',net.blobs[layer].data[0]) #[0] corresponds to the original image when oversample=False
        np.save(output_dir + 'input_image',net.deprocess('data', net.blobs['data'].data[0]))
    return net.blobs[layer].data[0]

def computeManyCNNFeatures(net, image_file, layers):
    # Create list that will be returned
    featureMap_list = []
    # Load image
    input_image = caffe.io.load_image(image_file)
    # Feed forward
    prediction = net.predict([input_image], oversample=False)
    # Store the desired feature maps
    for layer in layers:
        featureMap_list.append(net.blobs[layer].data[0])
    return featureMap_list


def predict(net, image_file, ov=False):
    input_image = caffe.io.load_image(image_file)
    prediction = net.predict([input_image], oversample=ov)
    return prediction
