'''
        NEGATIVE = 0
        POSITIVE = 1
'''


import numpy as np
import time, os, sys
import CNN_tools


# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
'''MODEL_FILE = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/test1/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_test1_iter_180.caffemodel'
    '''
MODEL_FILE = '/imatge/vcampos/work/flickr/80-20/prototxt/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/work/flickr/80-20/trained/Stage2/twitter_finetuned_iter_12000.caffemodel'
GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/twitter_five_agrees_absolutePath.txt'
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
SAVE_PATH = '/imatge/vcampos/work/twitter_finetuning/flickrFinetuned_results_5agrees/'
instanceList = []
correctLabels = 0
incorrectLabels = 0

CNN_tools.createDir(SAVE_PATH)

file = open(GROUND_TRUTH, "r")
false_positives = open(SAVE_PATH+"false_positives.txt","w") # images that are negative but were labeled as positive
false_negatives = open(SAVE_PATH+"false_negatives.txt","w") # images that are positive but were labeled as negative

# Store images in a list
while(True):
    line = file.readline()
    # Check if we have reached the end
    if (len(line)==0):
        break
    # Add the line to the list
    instanceList.append(line)


net = CNN_tools.loadNetCNN(MODEL_FILE, PRETRAINED, MEAN_FILE, (256,256), (2,1,0), 255)


# Loop through the ground truth file, predict each image's label and store the wrong ones
for instance in instanceList:
    values = instance.split()
    image_path = values[0]
    sentiment = int(values[1])
    prediction = CNN_tools.predict(net, image_path, True)
    # Check if the prediction was correct or not
    if prediction[0].argmax() == sentiment:
        correctLabels += 1
    else:
        incorrectLabels += 1
        # Write the image name (without path, so it can be used outside the servers) to the corresponding file
        if(prediction[0].argmax()==0):
            false_negatives.write(image_path.split('/')[-1]+"\n")
        else:
            false_positives.write(image_path.split('/')[-1]+"\n")


# Close files
file.close()
false_negatives.close()
false_positives.close()

# Print accuracy results
print 'Correct labels = ', str(correctLabels)
print 'Incorrect labels = ', str(incorrectLabels)
print 'Accuracy = ', str(100.*correctLabels/(correctLabels+incorrectLabels))
