import numpy as np
import time, os, sys
import CNN_tools

NEGATIVE = 0
POSITIVE = 1

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/test4/deploy.prototxt'
PRETRAINED = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_test4_iter_180.caffemodel'
#MODEL_FILE = '/imatge/vcampos/work/flickr/80-20/prototxt/deploy.prototxt'
#PRETRAINED = '/imatge/vcampos/work/flickr/80-20/trained/Stage2/twitter_finetuned_iter_12000.caffemodel'
GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/test4/test.txt'
MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
SAVE_PATH = '/imatge/vcampos/work/twitter_finetuning/topScores/test4/'


CNN_tools.createDir(SAVE_PATH)

file = open(GROUND_TRUTH, "r")
positiveList = open(SAVE_PATH+"positive_GT.txt","w")  # stores the images predicted as positive
negativeList = open(SAVE_PATH+"negative_GT.txt","w")  # stores the images predicted as negative

# Store images in a list
instanceList = []
while True:
    line = file.readline()
    # Check if we have reached the end
    if len(line) == 0:
        break
    # Add the line to the list
    instanceList.append(line)


# Load the CNN
net = CNN_tools.loadNetCNN(MODEL_FILE, PRETRAINED, MEAN_FILE, (256, 256), (2, 1, 0), 255)


# Loop through the ground truth file, predict each image's label and store the wrong ones
for instance in instanceList:
    values = instance.split()
    image_path = values[0]
    sentiment = int(values[1])
    prediction = CNN_tools.predict(net, image_path, True)
    # Check if the prediction was correct or not and write the image name (without path, so it can be used outside
    # the servers) to the corresponding file. Also attach the score and the ground truth
    score = prediction[0].item(prediction[0].argmax())
    s = image_path.split('/')[-1] + " " + str(sentiment) + " " + str(score) + "\n"
    if prediction[0].argmax() == POSITIVE:
        positiveList.write(s)
    else:
        negativeList.write(s)


# Close files
file.close()
negativeList.close()
positiveList.close()
