'''
        NEGATIVE = 0
        POSITIVE = 1
'''

import numpy as np
import time, os, sys
import CNN_tools


SAVE_DIR = '/imatge/vcampos/work/twitter_dataset/scores/'
SUBSETS = ['test1','test2','test3','test4','test5']
#MODEL_FILE = '/imatge/vcampos/work/twitter_finetuning/layer_removal/fc6/'+subset+'/deploy.prototxt'
#PRETRAINED = '/imatge/vcampos/work/twitter_finetuning/layer_removal/fc6/trained/twitter_finetuned_'+subset+'_iter_180.caffemodel'

def computeScores(path):
    # Function that computes scores of belonging to 'positive' class for the 5 folds
    # Returns (positiveScores, negativeScores) tuple
    positiveScores = []
    negativeScores = []
    for subset in SUBSETS:
        MODEL_FILE = path+subset+'/deploy.prototxt'
        PRETRAINED = path+'trained/twitter_finetuned_'+subset+'_iter_180.caffemodel'
        GROUND_TRUTH = '/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/'+subset+'/test.txt'
        MEAN_FILE = '/imatge/vcampos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
        instanceList = []

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


        # Loop through the ground truth file, predict each image's label and store its score
        for instance in instanceList:
            values = instance.split()
            image_path = values[0]
            sentiment = int(values[1])
            prediction = CNN_tools.predict(net, image_path, True)
            score = prediction[0].item(1)
            if sentiment == 0:
                negativeScores.append(score)
            else:
                positiveScores.append(score)

        # Close file
        file.close()

    print 'Prediction shape:' + str(prediction.shape)
    sys.stdout.flush()

    # Return scores
    return (positiveScores, negativeScores)



if __name__ == '__main__':
    ''' Create folder '''
    CNN_tools.createDir(SAVE_DIR)

    ''' Compute scores for each architecture '''
    # Regular fine-tuning
    print 'COMPUTING SCORES: Regular Fine-tuning...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/')
    CNN_tools.createDir(SAVE_DIR+'regular_fine-tuning/')
    np.save(SAVE_DIR+'regular_fine-tuning/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+'regular_fine-tuning/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (a)
    print 'COMPUTING SCORES: Architecture (a)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/layer_removal_1/fc7/')
    architecture_name = 'architecture_a'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (b)
    print 'COMPUTING SCORES: Architecture (b)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/layer_removal_1/fc6/')
    architecture_name = 'architecture_b'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (c)
    print 'COMPUTING SCORES: Architecture (c)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/layer_removal/fc7/')
    architecture_name = 'architecture_c'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (d)
    print 'COMPUTING SCORES: Architecture (d)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/layer_removal/fc6/')
    architecture_name = 'architecture_d'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (e)
    print 'COMPUTING SCORES: Architecture (e)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/layer_removal_1/fc8/')
    architecture_name = 'architecture_e'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()

    # Architecture (f)
    print 'COMPUTING SCORES: Architecture (f)...'
    sys.stdout.flush()
    [pos, neg] = computeScores('/imatge/vcampos/work/twitter_finetuning/fc9_5-fold_cross-validation/')
    architecture_name = 'architecture_f'
    CNN_tools.createDir(SAVE_DIR+architecture_name+'/')
    np.save(SAVE_DIR+architecture_name+'/'+'positiveScores',np.array(pos))
    np.save(SAVE_DIR+architecture_name+'/'+'negativeScores',np.array(neg))
    print 'Done\n\n'
    sys.stdout.flush()



