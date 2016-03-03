import os
import sys
import cv2
import numpy as np


if (len(sys.argv)>=2):
    try:
        model = str(sys.argv[1])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit("Not enough arguments. Run 'python compose_heatmaps.py model'")

PREDICTION_PATH = '/imatge/vcampos/work/twitter_dataset/fully_conv/'+model+'/predictions/'
IMAGES_PATH = '/imatge/vcampos/work/twitter_dataset/images/resized/'
OUTPUT_PATH = '/imatge/vcampos/work/twitter_dataset/fully_conv/'+model+'/composed_heatmaps/'

counter = 0
for path, dirs, files in os.walk(PREDICTION_PATH):
    for fileName in files:
        # Get path to the image
        image_name = fileName.split('.',2)[0] + '.jpg'
        image_path = os.path.join(IMAGES_PATH,image_name)

        # Load prediction
        prediction = np.load(os.path.join(path,fileName))
        heatmap = np.zeros((prediction.shape[1], prediction.shape[2], 3)) # BGR
        heatmap[:,:,1] = 255*prediction[1] # positive (1) in green
        heatmap[:,:,2] = 255*prediction[0] # negative (0) in red

        # Load image
        img = cv2.imread(image_path)
        if img is None:
		  pass

        # Resize heatmap so it fits the image
        heatmap = cv2.resize(heatmap, tuple(img.shape[1::-1]), interpolation=cv2.INTER_LINEAR)

        # Combine image and heatmap
        output = 0.5*img + 0.5*heatmap

        # Save result
        cv2.imwrite(os.path.join(OUTPUT_PATH,image_name), output)

        # Print progress
        counter += 1
        if counter%20 == 0:
            print 'Processed images: ' + str(counter)
            sys.stdout.flush()
