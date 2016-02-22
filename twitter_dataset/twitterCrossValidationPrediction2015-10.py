'''
        NEGATIVE = 0
        POSITIVE = 1
'''

import numpy as np
import time, os, sys

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/opt/caffe-2015-10/'
sys.path.insert(0, caffe_root + 'python')

import caffe


if (len(sys.argv)>=4):
    try:
        agrees = int(sys.argv[1])
        model = str(sys.argv[2])
        ov = int(sys.argv[3])
    except:
        sys.exit("The given arguments are not correct. Run 'python twitterCrossValidationPrediction2015-10 agrees model oversampling(0,1)'")
else:
    sys.exit("Not enough arguments. Run 'python twitterCrossValidationPrediction2015-10 agrees model oversampling(0,1)'")


if (agrees != 3) and (agrees != 4) and (agrees != 5):
    sys.exit("'agrees' must be 3, 4 or 5")


SUBSETS = ['test1','test2','test3','test4','test5']
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
ACCURACIES = []
OUTPUT_STRING = ""


if model == 'imagenet':
    if agrees == 3:
        model_folder = 'three_agree-5-fold_cross-validation'
    elif agrees == 4:
        model_folder = 'four_agree-5-fold_cross-validation'
    elif agrees == 5:
        model_folder = '5-fold_cross-validation'
elif model == 'places':
    MEAN_FILE = '/imatge/vcampos/caffe/models/places/places205CNN_mean.npy'
    model_folder = 'places_5-fold_CV'
elif model == 'deepsentibank':
    model_folder = 'deepsentibank_5-fold_CV'
elif model == 'mvso_en' or model == 'mvso_sp' or model == 'mvso_fr' or model == 'mvso_it' or model == 'mvso_ch' or model=='mvso_ge':
    model_folder = model + '_5-fold_CV'
elif model == 'fc9_mvso_en':
    model_folder = model + '_5-fold_CV'
else:
    sys.exit("The requested model is not valid")


if agrees == 3:
    iter = 255
    ground_truth_suffix = '_three_agrees'
elif agrees == 4:
    iter = 225
    ground_truth_suffix = '_four_agrees'
elif agrees == 5:
    iter = 180
    ground_truth_suffix = ''


if ov == 0:
    oversampling = False
elif ov == 1:
    oversampling = True
else:
    sys.exit("oversampling must be 0 or 1")


for subset in SUBSETS:
    deploy_path = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/'+subset+'/deploy.prototxt'
    caffemodel_path = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/trained/twitter_finetuned_'+subset+'_iter_'+str(iter)+'.caffemodel'
    GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation'+ground_truth_suffix+'/'+subset+'/test.txt'
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


    # Load network
    net = caffe.Classifier(deploy_path,
        caffemodel_path,
        mean=np.load(MEAN_FILE).mean(1).mean(1),
        image_dims=(256,256),
        channel_swap=(2,1,0), 
        raw_scale=255)


    # Loop through the ground truth file, predict each image's label and store the wrong ones
    counter = 0
    for instance in instanceList:
        values = instance.split()
        image_path = values[0]
        sentiment = int(values[1])

        # Load image
        im = caffe.io.load_image(image_path)

        # Make a forward pass and get the score
        prediction = net.predict([im], oversample=oversampling)

        #print '-------------------'
        #print prediction.shape
        #print '-------------------'
        #sys.stdout.flush()

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

        counter += 1
        if counter%40 == 0:
            print subset + ', ' + str(counter)
            sys.stdout.flush()


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
