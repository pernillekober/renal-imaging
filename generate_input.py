#!/usr/bin/env python
# coding: utf-8

# In[1]:


#open this notebook in environment by running following in terminal:
# source ~/miniconda/bin/activate
# jupyter notebook


# In[2]:


#!pip install nibabel
#!pip install matplotlib
#!pip install opencv-python


# In[3]:


#conda install --channel=conda-forge nilearn


# In[4]:


#For importing modules
import io, os, sys, types

#For file handling
from glob import glob
from os import listdir
from os.path import isfile, join
import json 

#For image processing
import PIL
import cv2
import nibabel as nib
import nibabel.processing 
from nibabel.processing import resample_to_output
import nilearn

#Image visualisation
import matplotlib.pyplot as plt

#Other
import numpy as np
import math
from random import sample

#Modeling
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


# ### Just some initial tests

# In[5]:


#!ls './preprocessed-data/images'


# In[6]:


#!ls './preprocessed-data/masks'


# In[7]:


#test_image=nib.load('./preprocessed-data/images/case_00004.nii.gz').get_fdata()

#print(test_image.shape, type(test_image))
#plt.imshow(test_image[2,:,:])


# ## Loading preprocessed images

# In[8]:


def malignant_labels_to_dict(json_file_path):
    """Takes path to json file and stores malignant label for each 
    patient in a dictionary"""
    
    with open(json_file_path) as user_file:
      file_contents = user_file.read()

    meta_list = json.loads(file_contents)

    labels_dict = {}
    for case in meta_list:
        c_id = case['case_id']
        labels_dict[c_id] = case['malignant']
    
    return labels_dict


# In[9]:


def generate_subsample_id_list(n_samples, pct_of_total_neg, dict_of_labels):
    """ generates a list of randomly samples case_ids with a number of negative labels 
    corresponding to the pct of total available negative cases, e.g 100% of 25 negative cases. """
    
    if n_samples > len(dict_of_labels):
        return "Number of subsamples must be less than {0}.".format(len(dict_of_labels))
    
    subsample_id_list = []
    total_neg = len(dict_of_labels) - sum(dict_of_labels.values())
    n_neg = math.ceil(total_neg*(pct_of_total_neg/100))
    n_pos = n_samples - n_neg
    
    true_list = [k for k,v in dict_of_labels.items() if v == True]
    false_list = [k for k,v in dict_of_labels.items() if v == False]
    
    subsample_id_list.extend(false_list[:n_neg])
    subsample_id_list.extend(sample(true_list,n_pos))
                             
    print("Generated list of {0} case IDs of {1} positive and {2} negative labels".format(
        len(subsample_id_list), len(true_list), len(false_list)))
    
    return subsample_id_list


# In[10]:


def processed_image_paths(rootdir, subfolder):
    """ Creates list of paths to images in each subfolder and corresponsing patient ID """
    
    folder_path = rootdir + subfolder
    paths_list = []
    for file in os.listdir(folder_path):
        paths_list.append(folder_path + '/' + file)
            
    return paths_list

def processed_partially_image_paths(rootdir, subfolder, list_of_cases):
    """ Creates list of paths to images in each subfolder and corresponsing patient ID """
    
    folder_path = rootdir + subfolder
    paths_list = []
    for file in os.listdir(folder_path):
        if file[:-4] in list_of_cases:
            paths_list.append(folder_path + '/' + file)
            
    return paths_list


# In[11]:


def load_nifti_img_and_mask_as_numpy(paths_list, subsample_list):
    """ Loads nifti images corresponding to paths lists in a dictionary 
    of case_id as keys and 3D numpy image arrays as values. Filters list by  """

    image_dict = {}

    for img_path in paths_list:
        case_id = img_path[-14:-4]
        if case_id in subsample_list:
            ct_nii = np.load(img_path)
            image_dict[case_id] = ct_nii
    
    return image_dict


# ## Creating traing, validation and test set

# In[12]:


def split_sets(cases, labels, val_pct, test_pct):
    
    temp_split = val_pct + test_pct
    test_split = test_pct/temp_split

    id_train, id_test_temp, labels_train, labels_test_temp = train_test_split(
        cases, labels, test_size=temp_split, shuffle=True, stratify=labels, random_state=42)

    id_val, id_test, labels_val, labels_test = train_test_split(
        id_test_temp, labels_test_temp, test_size=test_split, shuffle=True, random_state=42, stratify=labels_test_temp)
    
    print("Generated training, validation and test with {0}, {1} and {2} cases.".format(
        len(id_train), len(id_val), len(id_test)))
    
    return id_train, labels_train, id_val, labels_val, id_test, labels_test


# In[13]:


def fill_set(x_set, y_set, image_dictionary, label_dictionary):
    x_list, y_list = [], []
    
    for idx, (x, y) in enumerate(zip(x_set, y_set)):  
        for img in image_dictionary[x]:
            x_list.append(img)
            y_list.append(label_dictionary[x])
    
    x_array, y_array = np.array(x_list), np.array(y_list).astype(int)
    
    #x_array = np.array(tf.expand_dims(x_array, -1))
    y_array = np.asarray(y_array).astype('float32').reshape((-1,1))
    
    return x_array, y_array


# In[14]:


def adding_channel(x_array):
    
    x_array = np.repeat(x_array[..., np.newaxis], 3, -1)
    
    return x_array


# In[15]:


def generate_data_input(json_path, n_samples, neg_pct, rootdir, img_dir, mask_dir, val_split, test_split):

    #------------- Generate dictionary of case_id and binary malignant label --------
    labels_dict = malignant_labels_to_dict(json_path)
    print('Number of labels in labels_dict: ', len(labels_dict))

    #------------- Sample n instances from labels_dict with all possible negative cases --------
    subsample_list = generate_subsample_id_list(n_samples, neg_pct, labels_dict)

    #------------- Generate list of paths to preprocessed images --------

    image_paths_list = processed_image_paths(rootdir, img_dir)
    mask_paths_list = processed_image_paths(rootdir, mask_dir)
    print('Are image and mask path list the same lenght?: ', len(image_paths_list) == len(mask_paths_list))

    #------------- Load images and masks filtered by subsample --------
    image_dict = load_nifti_img_and_mask_as_numpy(image_paths_list, subsample_list)
    mask_dict = load_nifti_img_and_mask_as_numpy(mask_paths_list, subsample_list)
    print('Are image and mask dictionaries the same lenght?:', len(image_dict) == len(mask_dict))

    #--------------- flatten case and labels dictionary for split ---------------
    cases = subsample_list
    labels = [labels_dict[x] for x in cases]
    print("Are cases and labels lists the same length?: ", len(cases) == len(labels), len(cases), len(labels))

    #--------------- split training, validation and test set on patient/case level ---------------
    id_train, labels_train, id_val, labels_val, id_test, labels_test = split_sets(
        cases, labels, val_split, test_split)

    #--------------- fill training set with images ---------------
    x_train, y_train = fill_set(id_train, labels_train, image_dict, labels_dict)
    x_val, y_val = fill_set(id_val, labels_val, image_dict, labels_dict)
    x_test, y_test = fill_set(id_test, labels_test, image_dict, labels_dict)
    print("Are training set and labels lists the same length?: ", len(x_train) == len(y_train), len(x_train), len(y_train))

    #--------------- adding channel dimensions ---------------
    x_train, x_val, x_test = adding_channel(x_train), adding_channel(x_val), adding_channel(x_test)
    print('After adding channels: ', x_train.shape, x_val.shape, x_test.shape)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, id_val, id_test


# In[16]:


#x_train, y_train, x_val, y_val, x_test, y_test = generate_data_input(
#    '../kits21/kits21/data/kits.json', n_samples = 50, neg_pct = 100, rootdir = './preprocessed-data/', 
#    img_dir = 'images', mask_dir = 'masks', val_split = 10, test_split = 10)


# In[ ]:


#plt.imshow(x_train[400])

