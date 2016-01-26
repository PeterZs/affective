'''
        NEGATIVE = 0
        POSITIVE = 1
'''

import sys
import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/opt/caffe-2015-10/'
sys.path.insert(0, caffe_root + 'python')

import caffe


SUBSETS = ['test1','test2','test3','test4','test5']
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
#MEAN_FILE = '/imatge/vcampos/caffe/models/places/places205CNN_mean.npy'


for subset in SUBSETS:
    deploy_path = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/'+subset+'/deploy_conv.prototxt'
    caffemodel_path = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_'+subset+'_iter_180_conv.caffemodel'
    GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/'+subset+'/test.txt'

    file = open(GROUND_TRUTH,"r")

    # Store images in a list
    instanceList = []
    while(True):
        line = file.readline()
        # Check if we have reached the end
        if (len(line)==0):
            break
        # Add the line to the list
        instanceList.append(line)

    # Load fully convolutional network
    net_full_conv = caffe.Net(deploy_path, caffemodel_path, caffe.TEST)

    # Configure preprocessing
    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # Loop through the ground truth file, predict each image's label and store the wrong ones
    for instance in instanceList:
        # Get path and ground truth
        values = instance.split()
        image_path = values[0]
        sentiment = int(values[1])

        # Load image
        im = caffe.io.load_image(image_path)

        # Make a forward pass
        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        # Save the 2x8x8 prediction
        image_name = image_path.split('/')[-1]
        np.save('/imatge/vcampos/work/twitter_dataset/fully_conv/predictions/' + image_name.split('.',2)[0],
                out['prob'][0])

    file.close()

