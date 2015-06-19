import os, sys, time
import numpy as np
from sklearn.svm import SVC
import subprocess


#N_values = [-5, -3, -1, 0, 1, 3, 5, 7, 9] # C=2^N
N_values = [11,13,15]
LAYERS = ['fc8_twitter', 'fc7', 'fc6', 'pool5', 'conv5', 'conv4', 'conv3', 'norm2', 'pool2', 'conv2', 'norm1', 'pool1', 'conv1']
SUBSETS = ['test1','test2','test3','test4','test5']
SAVE_PATH = r"/imatge/vcampos/work/twitter_dataset/layer_analysis/"
SAVE_FILE = SAVE_PATH + "SVM_linear-3.txt"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print 'Create dir: ', SAVE_PATH

# Open the file where the results will be stored
results_file = open(SAVE_FILE, "w")
results_file.write("SVM results using C=2^N, with N = " + str(N_values)+"\n\n")

for LAYER_NAME in LAYERS:
    results_file.write("--------- Layer: " + LAYER_NAME + " ---------\n")
    print 'LAYER: ' + LAYER_NAME
    sys.stdout.flush()
    
    # Create vector where the accuracy for each C value will be accumulated
    accuracies = [0.0]*len(N_values)
    
    for subset in SUBSETS:
        print 'SUBSET: ' + subset
        sys.stdout.flush()
        
        results_file.write("\tSubset: " + subset + ":\n")
        TRAIN_GROUND_TRUTH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/" + subset + "/train.txt"
        TEST_GROUND_TRUTH = r"/imatge/vcampos/work/twitter_dataset/ground_truth/5-fold_cross-validation/" + subset + "/test.txt"
        FEATURES_PATH = r"/imatge/vcampos/work/twitter_dataset/feature_maps/" + subset + "/" + LAYER_NAME + "/"

        data_train = []
        data_test = []
        labels_train = []
        labels_test = []

        # Get the amount of lines in each file
        num_train_images = int( (subprocess.Popen( 'wc -l {0}'.format( TRAIN_GROUND_TRUTH ), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0] )
        num_test_images = int( (subprocess.Popen( 'wc -l {0}'.format( TEST_GROUND_TRUTH ), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0] )

        # Open files
        file_train = open(TRAIN_GROUND_TRUTH, "r")
        file_test = open(TEST_GROUND_TRUTH, "r")

        print 'Using feature maps from layer ' + LAYER_NAME
        print 'Loading feature maps...'
        sys.stdout.flush()

        # Load train data and ground truth
        t0 = time.time()
        counter = 0
        while(True):
            line = file_train.readline()
            # Check if we have reached the end
            if (len(line)==0):
                break
            # Add the line to the list
            values = line.split()
            labels_train.append(int(values[1]))
            imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
            feat = np.load(FEATURES_PATH+imageName+".npy")
            data_train.append(feat.flatten())
            # Update the counter and show progress every 300 iterations
            counter += 1
            if (counter%300 == 0):
                print 'TRAIN: Loaded ' + str(counter) + '/' + str(num_train_images) + ' feature maps'
                sys.stdout.flush() # without flushing, no prints are shown until the end
        print ("Loading train data: %.2f s" %(time.time()-t0))
        sys.stdout.flush()

        # Load test data and ground truth
        t0 = time.time()
        counter = 0
        while(True):
            line = file_test.readline()
            # Check if we have reached the end
            if (len(line)==0):
                break
            # Add the line to the list
            values = line.split()
            labels_test.append(int(values[1]))
            imageName = values[0].split("/")[-1][:-4] # takes the file name and removes the extension
            feat = np.load(FEATURES_PATH+imageName+".npy")
            data_test.append(feat.flatten())
            # Update the counter and show progress every 100 iterations
            counter += 1
            if (counter%100 == 0):
                print 'TEST: Loaded ' + str(counter) + '/' + str(num_test_images)  + ' feature maps'
                sys.stdout.flush() # without flushing, no prints are shown until the end
        print ("Loading test data: %.2f s" %(time.time()-t0))
        sys.stdout.flush()

        # Close files
        file_train.close()
        file_test.close()

        # Train an SVM for each value of N (C)
        i = 0
        for N in N_values:
            C = 2**N
            # Train SVM using train data
            clf = SVC(kernel='linear', C=C)
            print 'Training SVM with C='+str(C)+'...'
            sys.stdout.flush()
            t0 = time.time()
            clf.fit(data_train, labels_train)
            print ("Training SVM: %.2f s" %(time.time()-t0))
            sys.stdout.flush()

            # Test SVM
            print 'Testing SVM with C='+str(C)+'...'
            sys.stdout.flush()
            accuracy = 100.*clf.score(data_test, labels_test)
            print ("Accuracy = %.2f%%" %(accuracy))
            sys.stdout.flush()

            # Write result in results_file
            results_file.write("\t\tC = " + str(C) + "   Accuracy = " + str(accuracy) + "\n")

            # Store accuracy in the vector and increment i
            accuracies[i]+=accuracy
            i+=1

    # Store mean results for this layer
    s = "Mean results shown as (C,Acc):  "
    i = 0
    for N in N_values:
        s+="("+str(2**N)+","+str(accuracies[i]/len(SUBSETS))+")  "
        i+=1
    results_file.write(s+"\n")
    print s
    # Compute the best C for this layer
    best_i = np.argmax(accuracies)
    s = "\nBest classifier: C=" + str(2**N_values[best_i]) + ", Accuracy=" + str(accuracies[best_i]/len(SUBSETS)) + "\n\n"
    results_file.write(s)
    print s
    sys.stdout.flush()

# Close results file
results_file.close()