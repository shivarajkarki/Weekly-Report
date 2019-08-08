""" 
Created by Shivaraj J Karki, Emage Vision India 31st August 2019
"""

import numpy as np
import os
import cv2
import imutils
import time

#VGG16

def flip_image(input_list, orientation):
    
    """ 0, for flipping the image around the x-axis (horizontal flipping);
        1, for flipping around the y-axis (vertical flipping);
        -1, for horizontal mirror"""
        
    print("Flipping images...")
    
    #debug
    #print(orientation)
    
    if not orientation:
        #debug
        print('Flipping horizontal')
        #this one was flipping horizontal, so when rotating it gave same orientation as vertical
        flip = [0]
    elif orientation == 1:
        #debug
        print('Flipping vertical')
        flip = [1]
    else:
        #debug
        print('Flipping about both axis ')
        #so to get horizontal as well as mirror image, I'm using -1, 
        #in this case horizontal and vertical rotation give different orientation
        flip = [-1, 1]
    
    #flip the image
    return [cv2.flip(item,f) for item in input_list for f in flip]
    
def translate_image(input_list, pixel_lapse):
    
    #find shape of image
    img_height, img_width = input_list[0].shape
    
    #define set of matrices to translate 
    M0 = np.float32([[1,0,0],[0,1,0]]) #No translation... 
    M1 = np.float32([[1,0, pixel_lapse], [0,1, pixel_lapse]]) #linear translation... lower right corner
    M2 = np.float32([[1,0,-pixel_lapse], [0,1, pixel_lapse]]) #linear translation... lower left corner
    M3 = np.float32([[1,0, pixel_lapse], [0,1,-pixel_lapse]]) #linear translation... upper right corner
    M4 = np.float32([[1,0,-pixel_lapse], [0,1,-pixel_lapse]]) #linear translation... upper left corner
    
    #create a list of translator
    M = [M0, M1, M2, M3, M4]
    
    #debug purpose
    #M = [M0]
    
    print("Translating images, with ", pixel_lapse," pixel lapse")
    #translate images
    return [cv2.warpAffine(item, m, (img_width, img_height)) for item in input_list for m in M]

def rotate_image(input_list, smallest_angle):
    
    print("Rotating images with angle of: ", smallest_angle,)
    
    #rotates image with a gap of smallest_angle from 0 to 360 degree, better to take factor of 360
    return [imutils.rotate(item, angle) for item in input_list for angle in np.arange(0,360, smallest_angle)]

def augment_images(input_list,
                   flip = True,
                   flip_orientation = -1,
                   translate = True,
                   pixel_lapse = 120,
                   rotate = True,
                   smallest_rotate_angle = 15):
    
    """ This function calls Flip, Translate and Rotate Augmentation functions
    input_list is input list of images
    flip, translate, rotate are boolean and default enabled 
    flip_orientation 0-horizontal, 1-vertical, -1-both 
    pixel_lapse is no. of pixels to be shifted or translated
    smallest_rotate_angle is smallest angle to be rotates, a factor of 360"""
    
    if flip:
        flip_sequel = flip_image(input_list, flip_orientation)
    else:
        flip_sequel = input_list
        
    if translate:
        translated_sequel = translate_image(flip_sequel, pixel_lapse)
    else:
        translated_sequel = flip_sequel
        
    if rotate:
        rotated_sequel = rotate_image(translated_sequel, smallest_rotate_angle)
    else:
        rotated_sequel = translated_sequel
        
        
    return rotated_sequel
    

#Resize images
def resize_images(image_list, resize_width, resize_height):
    
    
    print("Maintaining aspect ratio...", float(image_list[0].shape[1]/image_list[0].shape[0]))
    
    resize_width = int(float(image_list[0].shape[1]/image_list[0].shape[0])*resize_height)
    
    #commented below code to alternate with list comprehension
    """
    resize_image_list = []
    for i, image in enumerate(image_list):
        image  = cv2.resize(image, (resize_width, resize_height))
        #debug
        #print(image.shape)
        resize_image_list.append(image)
    
    return resize_image_list
    """
    return [cv2.resize(image, (resize_width, resize_height)) for image in image_list]
        
    
    
#Now its time to write images from folder
    
def write_images_to_directory(new_dir, input_list, path):
    
    print("We have total ", len(input_list), " agumented images")
    
    #debug
    #print(path)
    #print(new_dir)
    
    if new_dir:
        path = path + '_new'
        os.mkdir(path)
        
        #debug
        print('Creating new Directory... . ',path)
        
    print("writing images inside: ", path," directory")
    
    # list comprehension takes long time to write
    
    for i, image in enumerate(input_list):
        cv2.imwrite(path+"/img"+str(i)+".bmp", image)
    """
    [cv2.imwrite(path+"/img"+str(i)+".bmp", image) for image in input_list for i in range(len(input_list))]
    """
    

def extract_images(directory):
    
    """ Read image files from given directory, and update in list
    Returns bmp images from given directory/folder"""
    cv_image = []
    
    #enable or desable image resize
    image_resize = True
    resize_width = 250
    resize_height = 250
    
    create_new_directory = True
    
    read_start = time.time()
    for r, d, f in os.walk(directory):
        #print for debug purpose
        print("\nReading Images Inside directory \"", directory,'\"\n')
        print("***********************************************************")
        
        for file in f:
            if '.bmp' in file:
                n = cv2.imread(directory+'\\'+file,0) #read gray scale image
                cv_image.append(n)
    #debug
    #print('Number of images in this directory: ',len(cv_image))
    
    print("\nReading done...we have total of ", len(cv_image), " images in ", directory,
          "\nTime taken to read these files: ", time.time()-read_start, " seconds")
    
    
    if len(cv_image) == 0:
        print("NO .BMP IMAGES... NO AUGMENTATION")
        print("***********************************************************")
        
    else:
        print("We have images of size: ", cv_image[0].shape)
        print("...Now Augmentation...\n")
        
        aug_time = time.time()
        
        augmented_sequel = augment_images(cv_image) 
        
        print("Agumentation done.... Now writing back to directory..",
             "\nTime taken for Augmenattion: ", time.time()-aug_time, " seconds")
        print('...')
        
        if image_resize:
            print("Resizing images..")
            augmented_sequel = resize_images(augmented_sequel, resize_width, resize_height)
            print('resized image size: ', augmented_sequel[0].shape)
        
        write_time = time.time()
        #need to write images back to same directory
        write_images_to_directory(create_new_directory, augmented_sequel, directory)
        
        print("Wrote back to files, \nTook ", time.time()-write_time," seconds to write")
        print("Seeing off from directory ", directory)
        print("***********************************************************")
    
def extract_directories():
    """This one will derive all the directories in current working directory
    Returns list of directories, extract them and initiate agumentation"""
    
    
    #debug purpose
    print('inside: ',os.getcwd())
    directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
    
    #print all directory names
    print("we have following diresctories: \n", directories)
    print()
    
    result = [extract_images(d) for d in directories]
    
    print()
    print("----------*************************-----------")
    print('Total number of diretories read...', len(result),'\n')
    

        
extract_directories()
