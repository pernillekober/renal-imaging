#!/usr/bin/env python
# coding: utf-8

# In[1]:


#open this notebook in environment by running following in terminal:
# source ~/miniconda/bin/activate
# jupyter notebook


# In[1]:


get_ipython().system('pip install nibabel')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install opencv-python')


# In[3]:


import numpy as np
import cv2
import math
from random import sample
from glob import glob
import os

import json 
import nibabel as nib
import nibabel.processing 
from nibabel.processing import resample_to_output
import nilearn

import matplotlib.pyplot as plt
import PIL
from PIL import Image

from os import listdir
from os.path import isfile, join

import tensorflow as tf
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from nilearn.image import concat_imgs, mean_img, resample_img

import tensorflow as tflow
from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[4]:


get_ipython().system("ls '../kits21/kits21/data/'")


# In[5]:


def generate_image_paths(rootdir_str, general_filename):
    """ Creates list of paths to images in each subfolder and corresponsing patient ID """
    
    paths_list = []
    
    for file in os.listdir(rootdir_str):
        d = os.path.join(rootdir_str, file)
        if os.path.isdir(d):
            paths_list.append(d + general_filename)
            
    return paths_list


# In[6]:


def load_nifti_img_and_mask_as_numpy(paths_list):
    """ Loads nifti images corresponding to paths lists in a dictionary 
    of case_id as keys and 3D numpy image arrays as values. Filters list by  """

    image_dict = {}

    for img_path in paths_list:
        case_id = img_path[22:32]
        ct_nii = nib.load(img_path).get_fdata()
        #ct_nii = np.load(img_path, allow_pickle=True)
        image_dict[case_id] = ct_nii
    
    return image_dict


#print(len(image_case_dict) == len(mask_case_dict))


# In[7]:


#identifying images to remove based on if they have kidney or tumor in the image.

def remove_images_without_kidney(dictionary_3D_images, dictionary_3D_masks):
    """Based on mask presence of segmentation values in mask images; identifies indexes of 
    empty masks and removes them from given images and masks dictionaries"""
    
    idx_remove_dict = {}
    
    #identifying position of images to remove from image dictionary
    for key, value in dictionary_3D_masks.items():
        idx_list = []
        
        #checks max value in mask in each 3D image volume and saves idexes of images to remove
        for idx, img in enumerate(value):
            if np.max(img) == 0.0:
                idx_list.append(idx)
                
        idx_remove_dict[key] = idx_list
        
    #removing images corresponding to "empty" mask images
    for key, value in idx_remove_dict.items():
        dictionary_3D_images[key] = np.delete(dictionary_3D_images[key], value, axis=0)
        dictionary_3D_masks[key] = np.delete(dictionary_3D_masks[key], value, axis=0)
    
    return dictionary_3D_images, dictionary_3D_masks


# In[8]:


#Preprocessing: downsample images and normalizing

def image_preprocessing(dictionary_3D_images):
    """ Downsamples images and masks to 224x224 and normalize pixel values 
    to range of 0 to 1."""
    
    for key, value in dictionary_3D_images.items():
        image_list = []
        
        for img in value:
            img_downsampled = cv2.resize(img, dsize=(128, 128)) #128x128 -> 224x224 -> 32*?
            normalized_image = cv2.normalize(img_downsampled, None, 0, 1, cv2.NORM_MINMAX)
            
            image_list.append(normalized_image)
            
        dictionary_3D_images[key] = np.array(image_list)
        
    return dictionary_3D_images

#images_dict_preprocessed = image_preprocessing(image_dict)
#masks_dict_preprocessed = image_preprocessing(mask_dict)


# In[9]:


def save_images(dir_name, img_dict):
    prep_folder = 'preprocessed-data'
    
    if not os.path.exists(prep_folder):
        os.makedirs(prep_folder)
    
    if not os.path.exists(prep_folder + '/' + dir_name):
        os.makedirs(prep_folder + '/' + dir_name)
    
    for key, value in img_dict.items():    
        print(key, value.shape)
        #ni_img = nib.Nifti1Image(value, affine=np.eye(4))
        #nib.save(ni_img, prep_folder +'/'+ dir_name + '/' + key + '.nii.gz')      
        np.save(prep_folder +'/'+ dir_name + '/' + key + '.npy', value)
        #ct_nii = np.load('./preprocessed-data/images/case_00011.npy')

    return None


# In[10]:


def chunk_process_images(rootdir, image_str, mask_str, chunk_size):
    
    paths_list_images = generate_image_paths(rootdir, image_str)
    paths_list_masks = generate_image_paths(rootdir, mask_str)
    
    paths_list_images = paths_list_images
    paths_list_masks = paths_list_masks
    
    image_dict_preprocessed, mask_dict_preprocessed = {}, {}
    #slices list into sublist of size chunk_size and remaining elements
    for i in range(0, len(paths_list_images), chunk_size):
        chunk_img = paths_list_images[i:i + chunk_size]
        chunk_mask = paths_list_masks[i:i + chunk_size]
        print(chunk_img)
        
        image_dict = load_nifti_img_and_mask_as_numpy(chunk_img)
        mask_dict = load_nifti_img_and_mask_as_numpy(chunk_mask)
        
        image_dict, mask_dict = remove_images_without_kidney(
            image_dict, mask_dict)
            
        image_dict_preprocessed_sub = image_preprocessing(image_dict)
        mask_dict_preprocessed_sub = image_preprocessing(mask_dict)
        
        save_images('images', image_dict_preprocessed_sub)
        save_images('masks', mask_dict_preprocessed_sub)
    
    return "Ran without error"


# In[11]:


#print(image_dict_preprocessed.keys())


# In[12]:


#img = image_dict_preprocessed['case_00141']
#print(test.shape)
#test = img[40,:,:]
#plt.imshow(test)


# In[13]:


# ----------- Creating image paths ----------


#image_paths_list = generate_image_paths(rootdir, '/imaging.nii.gz')
#mask_paths_list = generate_image_paths(rootdir, '/aggregated_AND_seg.nii.gz')

#print(len(image_paths_list) == len(mask_paths_list))

#image_case_dict = load_nifti_img_and_mask_as_numpy(image_paths_list[:20])
#mask_case_dict = load_nifti_img_and_mask_as_numpy(mask_paths_list[:20])


# ----------- Removing irrelevant images ----------
#image_dict, mask_dict = remove_images_without_kidney(image_case_dict, mask_case_dict) 


# ----------- Preprocessing and saving images ----------
rootdir = '../kits21/kits21/data/'
chunk_process_images(rootdir, '/imaging.nii.gz', '/aggregated_AND_seg.nii.gz', 10)


# In[14]:


#save_numpy_as_nifti('images', images_dict_preprocessed)


# In[15]:


get_ipython().system("ls './preprocessed-data/images/'")


# In[16]:


get_ipython().system("ls '../kits21/kits21/data/case_00179/'")


# In[17]:


ct_nii = np.load('./preprocessed-data/images/case_00176.npy')
print(ct_nii.shape)
test = ct_nii[10,:,:]
plt.imshow(test)


# ### Random test

# In[18]:


img = nib.load('./preprocessed-data/images/case_00112.nii.gz')
print(img.shape)
data = img.get_fdata()
print(data.shape)


def resize_image_stack_and_flip(data):
    diff = data.shape[1] - data.shape[0]
    l, h = math.floor(diff/2), math.ceil(diff/2)
    print(diff, l, h)

    #a = front_padding, data, back_padding
    if data.shape[0] < data.shape[1]:
        print('padding image from {0} to {1}'.format(data.shape[0], data.shape[1]))
        empty_array = np.zeros([data.shape[1], data.shape[2]])
        front_padding, back_padding = np.array([empty_array]*l), np.array([empty_array]*h)
        resized_img = np.concatenate((front_padding, data))
        resized_img = np.concatenate((resized_img,back_padding))

    if data.shape[0] >= data.shape[1]:
        resized_img = data[-l:h].copy()
        
    #xz_slice = data[:, data.shape[1] // 2 - 1, :]
    #zx_slice = resized_img[:, :, resized_img.shape[-1] // 2 - 1]
        
    return resized_img

resized_img = resize_image_stack_and_flip(data)
print(resized_img.shape)


# In[ ]:


middle_slice = resized_img[:, :, resized_img.shape[-1] // 2 - 1]
print(middle_slice.shape)
plt.imshow(middle_slice)
#plt.imshow(test)


# In[ ]:


yz_slice = data[img.shape[0] // 2 - 1, :, :]
print(yz_slice.shape)
plt.imshow(yz_slice)


# In[ ]:


xz_slice = resized_img[:, resized_img.shape[1] // 2 - 1, :]
print(xz_slice.shape)

plt.imshow(xz_slice)


# In[ ]:


ct_nii = nib.load('./preprocessed-data/images/case_00140.nii.gz').get_fdata()
print(ct_nii.shape)
test = ct_nii[180,:,:]
plt.imshow(test)


# In[ ]:


ct_nii = nib.load('./preprocessed-data/images/case_00070.nii.gz').get_fdata()
print(ct_nii.shape)
test = ct_nii[40,:,:]
plt.imshow(test)


# In[ ]:


ct_nii = nib.load('./preprocessed-data/images/case_00184.nii.gz').get_fdata()
print(ct_nii.shape)
test = ct_nii[20,:,:]
plt.imshow(test)


# In[ ]:


ct_nii = nib.load('../kits21/kits21/data/case_00179/aggregated_AND_seg.nii.gz').get_fdata()
print(ct_nii.shape)
test = ct_nii[40,:,:]
plt.imshow(test)


# In[ ]:




