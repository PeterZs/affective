import os
import sys
from PIL import Image, ImageChops, ImageOps
import CNN_tools


def resizeAndCenterCrop(f_in, f_out, size=(256,256), pad=False):
    
    try:
        image = Image.open(f_in).convert('RGB')
    except:
        print 'Error with ' + str(f_in)
        return -1
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size
    
    if pad:
        thumb = image.crop( (0, 0, size[0], size[1]) )
        
        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )
        
        thumb = ImageChops.offset(thumb, offset_x, offset_y)
    
    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))
    
    thumb.save(f_out)


def folder_resizeAndCenterCrop(imageFolder, newImageFolder, size=(256,256)):
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(imageFolder):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            print 'Resizing ' + str(os.path.join(path,fileName))
            resizeAndCenterCrop(os.path.join(path,fileName), os.path.join(newImageFolder,fileName),size)


def flickr_resizeAndCenterCrop(imageFolder, newImageFolder, size=(256,256)):
    imgExts = ["png", "bmp", "jpg"]
    list = os.listdir(imageFolder)
    for subfolder in list:
        if (os.path.isdir(imageFolder+subfolder)):
            print 'Resizing files in ' + imageFolder+subfolder
            CNN_tools.createDir(newImageFolder+subfolder)
            subfolder_list = os.listdir(imageFolder+subfolder)
            for file in subfolder_list:
                if(os.path.isfile(imageFolder+subfolder+"/"+file)):
                    ext = file[-3:].lower()
                    if ext not in imgExts:
                        continue
                    try:
                        resizeAndCenterCrop(os.path.join(imageFolder+subfolder,file), os.path.join(newImageFolder+subfolder,file),size)
                    except:
                        errorFile = open("/imatge/vcampos/work/flickr/errorLog.txt", "r+")
                        errorFile.write(os.path.join(imageFolder+subfolder,file))
                        errorFile.close()


'''def resize(folder, fileName, factor):
    filePath = os.path.join(folder, fileName)
    im = Image.open(filePath)
    w, h  = im.size
    newIm = im.resize((int(w*factor), int(h*factor)))
    # i am saving a copy, you can overrider orginal, or save to other folder
    newIm.save(filePath+"copy.png")


def bulkResize(imageFolder, factor):
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(imageFolder):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            
            resize(path, fileName, factor)
'''

if __name__ == "__main__":
    imageFolder=sys.argv[1] # first arg is path to image folder
    newImageFolder=sys.argv[2]
    new_size=int(sys.argv[3])
    CNN_tools.createDir(newImageFolder)
    #folder_resizeAndCenterCrop(imageFolder, newImageFolder, size=(new_size,new_size))
    flickr_resizeAndCenterCrop(imageFolder, newImageFolder, size=(new_size,new_size))

