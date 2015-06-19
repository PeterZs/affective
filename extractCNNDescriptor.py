import sys
import os
import time
import CNN_tools

IMAGE_FILE = '/imatge/vcampos/caffe/examples/images/cat.jpg'

if (len(sys.argv)>=2):
    try:
        IMAGE_FILE = str(sys.argv[1])
    except:
        sys.exit('The given arguments are not correct')
else:
    print 'Using ' + IMAGE_FILE

# Load CNN from ImageNet
net = CNN_tools.loadImageNetCNN()

# Compute the descriptor and save it
t0 = time.time()
CNN_tools.computeCNNFeatures(net, IMAGE_FILE, save_flag=True, output_dir='/imatge/vcampos/work/extracted_features/')
print 'Time spent = ' + str(time.time()-t0) + 's'