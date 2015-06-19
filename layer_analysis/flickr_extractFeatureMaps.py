import os, sys, time
import numpy as np
sys.path.insert(0, '/imatge/vcampos/work/pyxel/tools/affective')
import CNN_tools

''' GLOBAL VARIABLES '''
LAYER_NAME = 'fc7'
GROUND_TRUTH = "/imatge/vcampos/work/flickr/ground_truth_CC_256x256.txt"
MODEL_FILE = '/imatge/vcampos/work/flickr/80-20/prototxt/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/work/flickr/80-20/trained/Stage2/twitter_finetuned_iter_12000.caffemodel'
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
FIRST_IMAGE = 0 # avoids starting from the beginning if the process needs to be resumed from a certain point

''' OBTAIN THE LAYER TO EXTRACT AND CREATE A DIRECTORY TO STORE THE FEATURE MAPS '''
if (len(sys.argv)>=2):
    try:
        LAYER_NAME = str(sys.argv[1])
        if(len(sys.argv)>=3):
            FIRST_IMAGE = int(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit('Not enough arguments. Layer name expected.')

SAVE_PATH = '/imatge/vcampos/work/flickr/feature_maps/' + LAYER_NAME + '/'

CNN_tools.createDir(SAVE_PATH)


''' LOAD NET '''
net = CNN_tools.loadNetCNN(MODEL_FILE, PRETRAINED, MEAN_FILE, (256,256), (2,1,0), 255)


''' STORE IMAGES IN A LIST '''
file = open(GROUND_TRUTH, "r")
instanceList = []
t0 = time.time()
while(True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    instanceList.append(line.split()[0]) # store only the image path
file.close()
print ("Listing images: %.2f seconds" %(time.time()-t0))
sys.stdout.flush()


''' LOOP THROUGH THE LIST AND COMPUTE&SAVE FEATURE MAPS '''
t0 = time.time()
counter = 0
for path in instanceList:
    if(counter>=FIRST_IMAGE):
        # Obtain the file name (so features can be stored as fileName.npy)
        fileName = path.split('/')[-1][:-4]
        # Compute the feature map (CNN_tools manages everything)
        feat = CNN_tools.computeCNNFeatures(net, path, save_flag=False, layer=LAYER_NAME)
        # Save the feature map as a numpy array
        np.save(SAVE_PATH+fileName,feat)
        # Release memory
        del feat
    # Update counter and print information every 10 images
    counter += 1
    if(counter%10 == 0):
        print 'Progress: ' + str(counter) + '/' + str(len(instanceList))
        sys.stdout.flush() # without flushing, no prints are shown until the end


''' PRINT RESULTS '''
print 'Done. Time spent in feature extraction = ' + str((time.time()-t0)/60) + 'min'
print 'Feature maps extracted: ' + str(counter) + '/' + str(len(instanceList))
