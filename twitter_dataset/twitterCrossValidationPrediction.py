'''
        NEGATIVE = 0
        POSITIVE = 1
'''

import numpy as np
import time, os, sys
import CNN_tools


SUBSETS = ['test1','test2','test3','test4','test5']
ACCURACIES = []
OUTPUT_STRING = ""

for subset in SUBSETS:
    MODEL_FILE = '/imatge/vcampos/work/twitter_finetuning/places_5-fold_CV/'+subset+'/deploy.prototxt'
    PRETRAINED = '/imatge/vcampos/work/twitter_finetuning/places_5-fold_CV/trained/twitter_finetuned_'+subset+'_iter_180.caffemodel'
    GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/'+subset+'/test.txt'
    #MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    MEAN_FILE = '/imatge/vcampos/caffe/models/places/places205CNN_mean.npy'
    instanceList = []
    correctLabels = 0
    incorrectLabels = 0
    positiveLabels = 0
    negativeLabels = 0
    positivePredictions = 0
    negativePredictions = 0

    file = open(GROUND_TRUTH,"r")

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
        print '-------------------'
        print prediction.shape
        print '-------------------'
        sys.stdout.flush()
        # Check if the prediction was correct or not
        if prediction[0].argmax() == sentiment:
            correctLabels += 1
        else:
            incorrectLabels += 1
        # Update label counter
        if sentiment == 0:
            negativeLabels += 1
        else:
            positiveLabels += 1
        # Update prediction counter
        if prediction[0].argmax() == 0:
            negativePredictions += 1
        else:
            positivePredictions += 1


    file.close()
    accuracy = 100.*correctLabels/(correctLabels+incorrectLabels)
    ACCURACIES.append(accuracy)

    # Print accuracy results
    print '------------- ' + subset + ' -------------'
    print 'Accuracy = ', str(accuracy)
    print '---------------------------------'

    OUTPUT_STRING += "Subset: " + subset + ": " + "\n    Positive images: " + str(positiveLabels) + "\n    Negative images: " + str(negativeLabels) + "\n    Positive predictions: " + str(positivePredictions) + "\n    Negative predictions: " + str(negativePredictions) + "\n"


print '\nRESULTS:'
for i in range(0,5):
    print SUBSETS[i] + ': ' + str(ACCURACIES[i]) + '%'
print '\nMean accuracy = ' + str(1.*sum(ACCURACIES)/len(ACCURACIES))
print "\n-------------------------------------\n"
print OUTPUT_STRING
