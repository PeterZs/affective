import os, sys, time
import numpy as np
sys.path.insert(0, '/imatge/vcampos/work/pyxel/tools/affective')
import CNN_tools

''' GLOBAL VARIABLES '''
GROUND_TRUTH = "/imatge/vcampos/work/twitter_dataset/ground_truth/twitter_five_agrees_absolutePath.txt"
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
FIRST_IMAGE = 0 # avoids starting from the beginning if the process needs to be resumed from a certain point
LAYERS = ['pool5', 'norm2', 'pool2', 'norm1', 'pool1',]

''' OBTAIN THE SUBSET THAT IS BEING USED '''
if (len(sys.argv)>=2):
    try:
        SUBSET = str(sys.argv[1])
        if(len(sys.argv)>=3):
            FIRST_IMAGE = int(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit('Not enough arguments. Layer name expected.')

MODEL_FILE = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/test' + SUBSET + '/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_test' + SUBSET + '_iter_180.caffemodel'
SAVE_PATH = '/imatge/vcampos/work/twitter_dataset/feature_maps/test' + SUBSET + '/'




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


''' CREATE DIRECTORIES '''
CNN_tools.createDir(SAVE_PATH)
for layer in LAYERS:
    CNN_tools.createDir(SAVE_PATH+layer+'/')


''' LOOP THROUGH THE LIST AND COMPUTE&SAVE FEATURE MAPS '''
t0 = time.time()
counter = 0
for path in instanceList:
    if(counter>=FIRST_IMAGE):
        # Obtain the file name (so features can be stored as fileName.npy)
        fileName = path.split('/')[-1][:-4]
        # Compute the feature map (CNN_tools manages everything)
        feat_list = CNN_tools.computeManyCNNFeatures(net, path, LAYERS)
        i = 0
        for layer in LAYERS: # Save each feature map as a numpy array
            np.save(SAVE_PATH+layer+'/'+fileName,feat_list[i])
            i+=1
        # Release memory
    del feat_list[:]
    # Update counter and print information every 10 images
    counter += 1
    if(counter%10 == 0):
        print 'Progress: ' + str(counter) + '/' + str(len(instanceList))
        sys.stdout.flush() # without flushing, no prints are shown until the end




''' PRINT RESULTS '''
print 'Done. Time spent in feature extraction = ' + str((time.time()-t0)/60) + 'min'
print 'Feature maps extracted: ' + str(counter) + '/' + str(len(instanceList))
