import numpy as np
import sys

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/opt/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

MEAN_PATH = '/imatge/vcampos/caffe/models/places/'
FILE_NAME = 'places205CNN_mean'
input_file = MEAN_PATH + FILE_NAME + '.binaryproto'
output_file = MEAN_PATH + FILE_NAME + '.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( input_file , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
arr = arr.squeeze(0) # squeeze first axis
np.save(output_file, arr)
