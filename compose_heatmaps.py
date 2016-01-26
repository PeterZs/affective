import os
import cv2
import numpy as np
import CNN_tools

PREDICTION_PATH = '/imatge/vcampos/work/twitter_dataset/fully_conv/predictions/'
IMAGES_PATH = '/imatge/vcampos/work/twitter_dataset/images/resized/'
OUTPUT_PATH = '/imatge/vcampos/work/twitter_dataset/fully_conv/composed_heatmaps/'


# Create output directory
CNN_tools.createDir(OUTPUT_PATH)

counter = 0
for path, dirs, files in os.walk(PREDICTION_PATH):
    for fileName in files:
        # Get path to the image
        image_name = fileName.split('.',2)[0] + '.jpg'
        image_path = os.path.join(path,image_name)

        # Load prediction
        prediction = np.load(os.path.join(path,fileName))
        heatmap = np.zeros(3, prediction.shape[1], prediction.shape[2]) # BGR
        heatmap[1] = 255*prediction[0] # positive (0) in green
        heatmap[2] = 255*prediction[1] # negative (1) in red

        # Load image
        img = cv2.imread(image_path)

        # Resize heatmap so it fits the image
        heatmap = cv2.resize(heatmap, tuple(img.shape[1::-1]))

        # Combine image and heatmap
        output = 0.5*img + 0.5*heatmap

        # Save result
        cv2.imwrite(os.path.join(OUTPUT_PATH,image_name))

        # Print progress
        counter += 1
        if counter%20 == 0:
            print 'Processed images: ' + str(counter)
