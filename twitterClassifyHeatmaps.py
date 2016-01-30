import sys, shutil, os


def createDir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print 'Create dir: ', directory
    return directory

 

if __name__ == '__main__':   

	if (len(sys.argv)>=3):
	    try:
	        images_path = str(sys.argv[1])
	        if images_path[-1] != '/':
	        	images_path = images_path + '/'
	        text_files_path = str(sys.argv[2])
	    except:
	        sys.exit('The given arguments are not correct')
	else:
	    sys.exit("Not enough arguments. Run 'python twitterClassifyHeatmaps.py images_folder text_files_folder'")


	tp = open(text_files_path+'/TP.txt', "r")
	fp = open(text_files_path+'/FP.txt', "r")
	tn = open(text_files_path+'/TN.txt', "r")
	fn = open(text_files_path+'/FN.txt', "r")

	createDir(os.path.join(images_folder, 'TP'))
	createDir(os.path.join(images_folder, 'TN'))
	createDir(os.path.join(images_folder, 'FP'))
	createDir(os.path.join(images_folder, 'FN'))


	# Move images to the proper folder
	while(True):
        line = tp.readline()
        # Check if we have reached the end
        if (len(line)==0):
            break
        # Move image
        shutil.move(images_folder+line, os.path.join(images_folder, 'TP')+line)

    while(True):
        line = tn.readline()
        # Check if we have reached the end
        if (len(line)==0):
            break
        # Move image
        shutil.move(images_folder+line, os.path.join(images_folder, 'TN')+line)

    while(True):
        line = fp.readline()
        # Check if we have reached the end
        if (len(line)==0):
            break
        # Move image
        shutil.move(images_folder+line, os.path.join(images_folder, 'FP')+line)

    while(True):
        line = fn.readline()
        # Check if we have reached the end
        if (len(line)==0):
            break
        # Move image
        shutil.move(images_folder+line, os.path.join(images_folder, 'FN')+line)

    # Close files
    tp.close()
    tn.close()
    fp.close()
    fn.close()
