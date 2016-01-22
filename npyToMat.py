import sys
import os
import numpy as np
import scipy.io
import CNN_tools

if (len(sys.argv)>=3):
    try:
        npy_path = str(sys.argv[1])
        mat_path = str(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit("Not enough arguments. Run 'python input_path output_path'")


# Create output path
CNN_tools.createDir(mat_path)

# Compute t-sne
fileList = os.listdir(npy_path)
i = 0
for file in fileList:
    if file.endswith('.npy'):
        feat = np.load(npy_path+file)
        scipy.io.savemat(mat_path+file[:-4]+'.mat', mdict={'featureMap': feat})
        i = i + 1
        if (i%10 == 0):
            print 'PROGRESS: Converted ' + str(i) + ' files'
            sys.stdout.flush()
