
# coding: utf-8

# In[8]:

from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy
import pickle


# In[64]:

import keras
from keras import metrics
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, LeakyReLU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.models import model_from_yaml


# In[10]:


"""
Returns the tau's to be predicted
"""
def get_output_data(prefix, suffix = 'T1'):
    tau_11 = loadmat(prefix + 'tau11_F_xyz_' + suffix + '.mat')['tau11']
    tau_12 = loadmat(prefix + 'tau12_F_xyz_' + suffix + '.mat')['tau12']
    tau_13 = loadmat(prefix + 'tau13_F_xyz_' + suffix + '.mat')['tau13']
    tau_22 = loadmat(prefix + 'tau22_F_xyz_' + suffix + '.mat')['tau22']
    tau_23 = loadmat(prefix + 'tau23_F_xyz_' + suffix + '.mat')['tau23']
    tau_33 = loadmat(prefix + 'tau33_F_xyz_' + suffix + '.mat')['tau33']
    
    tau_11 = np.pad(tau_11, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    tau_12 = np.pad(tau_12, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    tau_13 = np.pad(tau_13, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    tau_22 = np.pad(tau_22, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    tau_23 = np.pad(tau_23, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    tau_33 = np.pad(tau_33, ((1, 1), (1, 1), (1, 1)), 'constant', 
                    constant_values=((0, 0), (0, 0), (0, 0)))
    
    return tau_11, tau_12, tau_13, tau_22, tau_23, tau_33


# In[11]:


"""
Returns the tau's to be predicted
"""
def pad_and_filter(data):
    
    dims = data.shape
    
    data_augmented = np.concatenate([np.concatenate([data, data, data], axis = 0),
                                     np.concatenate([data, data, data], axis = 0),
                                     np.concatenate([data, data, data], axis = 0)],
                                    axis = 1)
    
    augmented_dims = data_augmented.shape
    
    data_filter = data_augmented[int((augmented_dims[0] - dims[0] - 2)/2):int((augmented_dims[0] + dims[0] + 2)/2),
                                 int((augmented_dims[1] - dims[1] - 2)/2):int((augmented_dims[1] + dims[1] + 2)/2),
                                 :]
    
    data_final = np.pad(data_filter, ((0, 0), (0, 0), (1, 1)), 'constant', 
                        constant_values=((0, 0), (0, 0), (0, 0)))
    
    return data_final


# In[12]:


"""
Returns the tau's to be predicted
"""
def get_input_data(prefix, suffix = 'T1'):
    uf = loadmat(prefix + 'u_F_xyz_' + suffix + '.mat')['u_F']
    vf = loadmat(prefix + 'v_F_xyz_' + suffix + '.mat')['v_F']
    wf = loadmat(prefix + 'w_F_xyz_' + suffix + '.mat')['w_F']
    tke = loadmat(prefix + 'TKE_F_xyz_' + suffix + '.mat')['TKE_F']
    theta = loadmat(prefix + 'theta_F_xyz_' + suffix + '.mat')['theta_F']
    
    uf = pad_and_filter(uf)
    vf = pad_and_filter(vf)
    wf = pad_and_filter(wf)
    tke = pad_and_filter(tke)
    theta = pad_and_filter(theta)

    return uf, vf, wf, tke, theta


# In[13]:


"""
Denormalizes Outputs
"""
def denormalize_data_feng(train, valid, test):
    mu, std = np.mean(train.flatten()), np.std(train.flatten())
    train_new = (train - mu)/std
    valid_new = (valid - mu)/std
    test_new = (test - mu)/std
    print()
    return train_new, valid_new, test_new, mu, std


# In[14]:


"""
Denormalizes Outputs
"""
def denormalize_data_feng_mu_std(train, test, mu, std):
    #mu, std = np.mean(train.flatten()), np.std(train.flatten())
    train_new = (train - mu)/std
    test_new = (test - mu)/std
    print()
    return train_new, test_new


# In[75]:


"""
Reshapes Data and split data into train, validation and test sets (Convolutional & Feng Set up)
"""
def create_train_test_sets_conv3d_feng(tau_11, tau_12, tau_13, tau_22, tau_23, tau_33,
                                       uf, vf, wf, tke, theta, #grad,
                                       train_pct, size = 3, augmentation = None):
    
    train_index = np.concatenate((np.ones((int(uf.shape[0]*1/2), uf.shape[1], uf.shape[2]), dtype = 'bool'),
                                  np.zeros((int(uf.shape[0]*1/2), uf.shape[1], uf.shape[2]), dtype = 'bool')),
                                 axis = 0)
    valid_index = np.concatenate((np.zeros((int(uf.shape[0]*1/2), uf.shape[1], uf.shape[2]), dtype = 'bool'),
                                  np.ones((int(uf.shape[0]*1/4), uf.shape[1], uf.shape[2]), dtype = 'bool'),
                                  np.zeros((int(uf.shape[0]*1/4), uf.shape[1], uf.shape[2]), dtype = 'bool')),
                                 axis = 0)
    test_index = np.concatenate((np.zeros((int(uf.shape[0]*3/4), uf.shape[1], uf.shape[2]), dtype = 'bool'),
                                 np.ones((int(uf.shape[0]*1/4), uf.shape[1], uf.shape[2]), dtype = 'bool')),
                                 axis = 0)
    
    print(train_index.shape)
    
    offset_size = int(size/2)
    
    for idx in range(8):
        
        scaled_idx = int(idx * uf.shape[0]/8)
        scaled_idx_end = int((idx + 1) * uf.shape[0]/8)
        
        train_index[scaled_idx:(scaled_idx+offset_size),:,:] = False
        train_index[(scaled_idx_end-offset_size):(scaled_idx_end),:,:] = False
        train_index[:,0:(0+offset_size),:] = False
        train_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
        train_index[:,:,0:(0+offset_size)] = False
        train_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
        
        test_index[scaled_idx:(scaled_idx+offset_size),:,:] = False
        test_index[(scaled_idx_end-offset_size):(scaled_idx_end),:,:] = False
        test_index[:,0:(0+offset_size),:] = False
        test_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
        test_index[:,:,0:(0+offset_size)] = False
        test_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
        
        valid_index[scaled_idx:(scaled_idx+offset_size),:,:] = False
        valid_index[(scaled_idx_end-offset_size):(scaled_idx_end),:,:] = False
        valid_index[:,0:(0+offset_size),:] = False
        valid_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
        valid_index[:,:,0:(0+offset_size)] = False
        valid_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
    
    train_index[:,:,55:] = False
    valid_index[:,:,55:] = False
    test_index[:,:,55:] = False
    
    train_locs = np.where(train_index)
    valid_locs = np.where(valid_index)
    test_locs = np.where(test_index)
    
    print(train_locs)
    
    tau_11_train, tau_11_valid, tau_11_test = np.transpose([tau_11[train_locs]]), np.transpose([tau_11[valid_locs]]), np.transpose([tau_11[test_locs]])
    tau_12_train, tau_12_valid, tau_12_test = np.transpose([tau_12[train_locs]]), np.transpose([tau_12[valid_locs]]), np.transpose([tau_12[test_locs]])
    tau_13_train, tau_13_valid, tau_13_test = np.transpose([tau_13[train_locs]]), np.transpose([tau_13[valid_locs]]), np.transpose([tau_13[test_locs]])
    tau_22_train, tau_22_valid, tau_22_test = np.transpose([tau_22[train_locs]]), np.transpose([tau_22[valid_locs]]), np.transpose([tau_22[test_locs]])
    tau_23_train, tau_23_valid, tau_23_test = np.transpose([tau_23[train_locs]]), np.transpose([tau_23[valid_locs]]), np.transpose([tau_23[test_locs]])
    tau_33_train, tau_33_valid, tau_33_test = np.transpose([tau_33[train_locs]]), np.transpose([tau_33[valid_locs]]), np.transpose([tau_33[test_locs]])
    
    tau_11_train, tau_11_valid, tau_11_test, mu_11, std_11 = denormalize_data_feng(tau_11_train, tau_11_valid, tau_11_test)
    tau_12_train, tau_12_valid, tau_12_test, mu_12, std_12 = denormalize_data_feng(tau_12_train, tau_12_valid, tau_12_test)
    tau_13_train, tau_13_valid, tau_13_test, mu_13, std_13 = denormalize_data_feng(tau_13_train, tau_13_valid, tau_13_test)
    tau_22_train, tau_22_valid, tau_22_test, mu_22, std_22 = denormalize_data_feng(tau_22_train, tau_22_valid, tau_22_test)
    tau_23_train, tau_23_valid, tau_23_test, mu_23, std_23 = denormalize_data_feng(tau_23_train, tau_23_valid, tau_23_test)
    tau_33_train, tau_33_valid, tau_33_test, mu_33, std_33 = denormalize_data_feng(tau_33_train, tau_33_valid, tau_33_test)
    
    x_train = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  vf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  wf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                 ], 
                                 axis = 3)
              for x,y,z in zip(train_locs[0], train_locs[1], train_locs[2])])
    
    x_valid = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  vf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  wf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                 ], 
                                 axis = 3)
              for x,y,z in zip(valid_locs[0], valid_locs[1], valid_locs[2])])

    x_test = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 vf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 wf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                ], 
                                axis = 3)
              for x,y,z in zip(test_locs[0], test_locs[1], test_locs[2])])
    
    print(x_train.shape)
    x_normalized = [denormalize_data_feng(x_train[:,:,:,:,k], x_valid[:,:,:,:,k], x_test[:,:,:,:,k]) 
                    for k in range(x_train.shape[4])]
    x_train = np.array(np.stack([k[0] for k in x_normalized], axis = 4))
    x_valid = np.array(np.stack([k[1] for k in x_normalized], axis = 4))
    x_test = np.array(np.stack([k[2] for k in x_normalized], axis = 4)) 
    
    x_mu = np.array([k[3] for k in x_normalized])
    x_std = np.array([k[4] for k in x_normalized])        
    
    print('X_train shape', x_train.shape)
    print('tau_train shape', tau_11_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    return (x_train, x_valid, x_test, tau_11_train, tau_11_valid, tau_11_test, 
            tau_12_train, tau_12_valid, tau_12_test, tau_13_train, tau_13_valid, tau_13_test,
           tau_22_train, tau_22_valid, tau_22_test, tau_23_train, tau_23_valid, tau_23_test, 
            tau_33_train, tau_33_valid, tau_33_test,
           mu_11, std_11, mu_12, std_12, mu_13, std_13, mu_22, std_22, mu_23, std_23, mu_33, std_33, x_mu, x_std)


# In[76]:


"""
Reshapes Data and split data into train, validation and test sets (Convolutional & Feng Set up)
"""
def create_train_test_sets_conv3d_feng_test(tau_11, tau_12, tau_13, tau_22, tau_23, tau_33,
                                            uf, vf, wf, tke, theta, #grad,
                                            train_pct, size = 3, augmentation = None):
    
    train_index = np.concatenate((np.zeros((uf.shape[0], 0, uf.shape[2]), dtype = 'bool'),
                                  np.ones((uf.shape[0], uf.shape[1], uf.shape[2]), dtype = 'bool')),
                                  axis = 1)
    valid_index = np.concatenate((np.ones((uf.shape[0], uf.shape[1], uf.shape[2]), dtype = 'bool'),
                                  np.zeros((uf.shape[0], 0, uf.shape[2]), dtype = 'bool')),
                                  axis = 1)
    
    test_index = np.ones((uf.shape[0], uf.shape[1], uf.shape[2]), dtype = 'bool')
    
    print(train_index.shape)
    
    offset_size = int(size/2)
    
    train_index[0:(0+offset_size),:,:] = False
    train_index[(uf.shape[0]-offset_size):(uf.shape[0]),:,:] = False
    train_index[:,0:(0+offset_size),:] = False
    train_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
    train_index[:,:,0:(0+offset_size)] = False
    train_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
    test_index[0:(0+offset_size),:,:] = False
    test_index[(uf.shape[0]-offset_size):(uf.shape[0]),:,:] = False
    test_index[:,0:(0+offset_size),:] = False
    test_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
    test_index[:,:,0:(0+offset_size)] = False
    test_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
    valid_index[0:(0+offset_size),:,:] = False
    valid_index[(uf.shape[0]-offset_size):(uf.shape[0]),:,:] = False
    valid_index[:,0:(0+offset_size),:] = False
    valid_index[:,(uf.shape[1]-offset_size):(uf.shape[1]),:] = False
    valid_index[:,:,0:(0+offset_size)] = False
    valid_index[:,:,(uf.shape[2]-offset_size):(uf.shape[2])] = False
    train_index[:,:,int(uf.shape[2]*3/4):] = False
    valid_index[:,:,int(uf.shape[2]*3/4):] = False
    test_index[:,:,int(uf.shape[2]*3/4):] = False
    
    train_locs = np.where(train_index)
    valid_locs = np.where(valid_index)
    test_locs = np.where(test_index)
    
    print(train_locs)
    
    tau_11_train, tau_11_valid, tau_11_test = np.transpose([tau_11[train_locs]]), np.transpose([tau_11[valid_locs]]), np.transpose([tau_11[test_locs]])
    tau_12_train, tau_12_valid, tau_12_test = np.transpose([tau_12[train_locs]]), np.transpose([tau_12[valid_locs]]), np.transpose([tau_12[test_locs]])
    tau_13_train, tau_13_valid, tau_13_test = np.transpose([tau_13[train_locs]]), np.transpose([tau_13[valid_locs]]), np.transpose([tau_13[test_locs]])
    tau_22_train, tau_22_valid, tau_22_test = np.transpose([tau_22[train_locs]]), np.transpose([tau_22[valid_locs]]), np.transpose([tau_22[test_locs]])
    tau_23_train, tau_23_valid, tau_23_test = np.transpose([tau_23[train_locs]]), np.transpose([tau_23[valid_locs]]), np.transpose([tau_23[test_locs]])
    tau_33_train, tau_33_valid, tau_33_test = np.transpose([tau_33[train_locs]]), np.transpose([tau_33[valid_locs]]), np.transpose([tau_33[test_locs]])
    
    tau_11_train_2, tau_11_valid_2, tau_11_test_2, mu_11, std_11 = denormalize_data_feng(tau_11_train, tau_11_valid, tau_11_test)
    tau_12_train_2, tau_12_valid_2, tau_12_test_2, mu_12, std_12 = denormalize_data_feng(tau_12_train, tau_12_valid, tau_12_test)
    tau_13_train_2, tau_13_valid_2, tau_13_test_2, mu_13, std_13 = denormalize_data_feng(tau_13_train, tau_13_valid, tau_13_test)
    tau_22_train_2, tau_22_valid_2, tau_22_test_2, mu_22, std_22 = denormalize_data_feng(tau_22_train, tau_22_valid, tau_22_test)
    tau_23_train_2, tau_23_valid_2, tau_23_test_2, mu_23, std_23 = denormalize_data_feng(tau_23_train, tau_23_valid, tau_23_test)
    tau_33_train_2, tau_33_valid_2, tau_33_test_2, mu_33, std_33 = denormalize_data_feng(tau_33_train, tau_33_valid, tau_33_test)
    
    x_train = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  vf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  wf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                 ], 
                                 axis = 3)
              for x,y,z in zip(train_locs[0], train_locs[1], train_locs[2])])
    
    x_valid = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  vf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  wf[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                 ], 
                                 axis = 3)
              for x,y,z in zip(valid_locs[0], valid_locs[1], valid_locs[2])])

    x_test = np.array([np.stack([uf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 vf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 wf[(x-offset_size):(x+offset_size+1),
                                    (y-offset_size):(y+offset_size+1),
                                    (z-offset_size):(z+offset_size+1)],
                                 tke[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)],
                                  theta[(x-offset_size):(x+offset_size+1),
                                     (y-offset_size):(y+offset_size+1),
                                     (z-offset_size):(z+offset_size+1)]#,
                                  #grad[(x-offset_size):(x+offset_size+1),
                                  #   (y-offset_size):(y+offset_size+1),
                                  #   (z-offset_size):(z+offset_size+1)]
                                ], 
                                axis = 3)
              for x,y,z in zip(test_locs[0], test_locs[1], test_locs[2])])
    
    x_normalized = [denormalize_data_feng(x_train[:,:,:,:,k], x_valid[:,:,:,:,k], x_test[:,:,:,:,k]) 
                    for k in range(x_train.shape[4])]
    x_train = np.array(np.stack([k[0] for k in x_normalized], axis = 4))
    x_valid = np.array(np.stack([k[1] for k in x_normalized], axis = 4))
    x_test = np.array(np.stack([k[2] for k in x_normalized], axis = 4)) 
    
    x_mu = np.array([k[3] for k in x_normalized])
    x_std = np.array([k[4] for k in x_normalized])        
    
    print('X_train shape', x_train.shape)
    print('tau_train shape', tau_11_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    return (x_train, x_valid, x_test, tau_11_train, tau_11_valid, tau_11_test, 
            tau_12_train, tau_12_valid, tau_12_test, tau_13_train, tau_13_valid, tau_13_test,
           tau_22_train, tau_22_valid, tau_22_test, tau_23_train, tau_23_valid, tau_23_test, 
            tau_33_train, tau_33_valid, tau_33_test,
           mu_11, std_11, mu_12, std_12, mu_13, std_13, mu_22, std_22, mu_23, std_23, mu_33, std_33, x_mu, x_std)


# In[77]:


"""
Trains Two-Layer Neural Network with Relu Activation Functions
"""
def train_conv_3d_model_feng(x_train, x_valid, x_test, y_train, y_valid, y_test, act_func = 'tanh',
                          batch_size = 1024, epochs = 20, num_nodes = 6, xdim = 3, size = 3):
    
    if act_func == 'relu':
        final_act_func = 'linear'
    else:
        final_act_func = act_func
        
    model = Sequential()
    
    model.add(Conv3D(128, kernel_size = (size,size,size), data_format = 'channels_last',
                         input_shape = x_train[...,:xdim].shape[1:], kernel_initializer = 'random_uniform'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1, activation = final_act_func))
    

    model.summary()
    
    callback = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')]

    model.compile(loss='mse',
                  optimizer=SGD(),
                  metrics=[metrics.mse])

    history = model.fit(x_train[...,:xdim], y_train[...,:xdim],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_valid[...,:xdim], y_valid[...,:xdim]),
                        callbacks = callback)
    
    score = model.evaluate(x_test[...,:xdim], y_test[...,:xdim], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return model


# In[78]:


"""
Plots Actual vs. Predicted Values from Model
"""
def visualize(model, x_test, y_test, mu, std):
    y_predict = model.predict(x_test)
    sample_index = (np.random.rand(y_test.shape[0]) < 1000./y_test.shape[0])
    plt.figure(figsize=(15,5))
    plt.plot(y_test[sample_index]*std+mu)
    plt.plot(y_predict[sample_index]*std+mu)
    plt.show()


# In[79]:


models = {'conv-3d NN Feng': (create_train_test_sets_conv3d_feng, train_conv_3d_model_feng),
          'conv-3d NN Feng Test': (create_train_test_sets_conv3d_feng_test, train_conv_3d_model_feng)}


# In[80]:


"""
Plots Actual vs. Predicted Values from Model
"""
def predict(model, x_test, y_test, mu, std):
    y_test = y_test.flatten()*std + mu
    y_predict = model.predict(x_test).flatten()*std + mu
    return np.corrcoef(y_test, y_predict), np.sqrt(((y_test - y_predict) ** 2).mean())


# In[81]:


"""
Main Function to Execute Model
"""
def main_feng(model_name, size = 1, augmentation = None, xdim = 6,  prefix = '', train_pct = 0.5):
    
    # Output Data
    tau_11_1, tau_12_1, tau_13_1, tau_22_1, tau_23_1, tau_33_1 = get_output_data('Re15sh20_DNScoarse8/', suffix = 'T1')
    tau_11_2, tau_12_2, tau_13_2, tau_22_2, tau_23_2, tau_33_2 = get_output_data('Re15sh20_DNScoarse8_T2/', suffix = 'T2')
    tau_11_3, tau_12_3, tau_13_3, tau_22_3, tau_23_3, tau_33_3 = get_output_data('Re15sh20_DNScoarse8_T3/', suffix = 'T3')
    tau_11_4, tau_12_4, tau_13_4, tau_22_4, tau_23_4, tau_33_4 = get_output_data('Re15sh20_DNScoarse8_T4/', suffix = 'T4')
    tau_11_5, tau_12_5, tau_13_5, tau_22_5, tau_23_5, tau_33_5 = get_output_data('Re15sh20_DNScoarse8_T5/', suffix = 'T5')
    tau_11_6, tau_12_6, tau_13_6, tau_22_6, tau_23_6, tau_33_6 = get_output_data('Re15sh20_DNScoarse8_T6/', suffix = 'T6')
    tau_11_7, tau_12_7, tau_13_7, tau_22_7, tau_23_7, tau_33_7 = get_output_data('Re15sh20_DNScoarse8_T7/', suffix = 'T7')
    tau_11_8, tau_12_8, tau_13_8, tau_22_8, tau_23_8, tau_33_8 = get_output_data('Re15sh20_DNScoarse8_T8/', suffix = 'T8')
    
    tau_11 = np.concatenate([tau_11_1, tau_11_2, tau_11_3, tau_11_4, tau_11_5, tau_11_6, tau_11_7, tau_11_8],
                            axis = 0)
    tau_12 = np.concatenate([tau_12_1, tau_12_2, tau_12_3, tau_12_4, tau_12_5, tau_12_6, tau_12_7, tau_12_8],
                            axis = 0)
    tau_13 = np.concatenate([tau_13_1, tau_13_2, tau_13_3, tau_13_4, tau_13_5, tau_13_6, tau_13_7, tau_13_8],
                            axis = 0)
    tau_22 = np.concatenate([tau_22_1, tau_22_2, tau_22_3, tau_22_4, tau_22_5, tau_22_6, tau_22_7, tau_22_8],
                            axis = 0)
    tau_23 = np.concatenate([tau_23_1, tau_23_2, tau_23_3, tau_23_4, tau_23_5, tau_23_6, tau_23_7, tau_23_8],
                            axis = 0)
    tau_33 = np.concatenate([tau_33_1, tau_33_2, tau_33_3, tau_33_4, tau_33_5, tau_33_6, tau_33_7, tau_33_8],
                            axis = 0)
    
    print('Shape of Output Files:')
    print(tau_11.shape, tau_12.shape, tau_13.shape, tau_22.shape, tau_23.shape, tau_33.shape)
    
    # Input Data
    uf_1, vf_1, wf_1, tke_1, theta_1 = get_input_data('Re15sh20_DNScoarse8/', suffix = 'T1')
    uf_2, vf_2, wf_2, tke_2, theta_2 = get_input_data('Re15sh20_DNScoarse8_T2/', suffix = 'T2')
    uf_3, vf_3, wf_3, tke_3, theta_3 = get_input_data('Re15sh20_DNScoarse8_T3/', suffix = 'T3')
    uf_4, vf_4, wf_4, tke_4, theta_4 = get_input_data('Re15sh20_DNScoarse8_T4/', suffix = 'T4')
    uf_5, vf_5, wf_5, tke_5, theta_5 = get_input_data('Re15sh20_DNScoarse8_T5/', suffix = 'T5')
    uf_6, vf_6, wf_6, tke_6, theta_6 = get_input_data('Re15sh20_DNScoarse8_T6/', suffix = 'T6')
    uf_7, vf_7, wf_7, tke_7, theta_7 = get_input_data('Re15sh20_DNScoarse8_T7/', suffix = 'T7')
    uf_8, vf_8, wf_8, tke_8, theta_8 = get_input_data('Re15sh20_DNScoarse8_T8/', suffix = 'T8')
    
    uf = np.concatenate([uf_1, uf_2, uf_3, uf_4, uf_5, uf_6, uf_7, uf_8], 
                         axis = 0)
    vf = np.concatenate([vf_1, vf_2, vf_3, vf_4, vf_5, vf_6, vf_7, vf_8], 
                         axis = 0)
    wf = np.concatenate([wf_1, wf_2, wf_3, wf_4, wf_5, wf_6, wf_7, wf_8], 
                         axis = 0)
    tke = np.concatenate([tke_1, tke_2, tke_3, tke_4, tke_5, tke_6, tke_7, tke_8], 
                         axis = 0)
    theta = np.concatenate([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8], 
                         axis = 0)
    
    # Get Functions
    train_test_split_func, model_func = models[model_name]
    
    # Reshape Data and Get Train/Test Sets
    (x_train, x_valid, x_test, tau_11_train, tau_11_valid, tau_11_test, 
            tau_12_train, tau_12_valid, tau_12_test, tau_13_train, tau_13_valid, tau_13_test,
           tau_22_train, tau_22_valid, tau_22_test, tau_23_train, tau_23_valid, tau_23_test, 
            tau_33_train, tau_33_valid, tau_33_test,
           mu_11, std_11, mu_12, std_12, mu_13, std_13, 
     mu_22, std_22, mu_23, std_23, mu_33, std_33, x_mu, x_std) = train_test_split_func(tau_11, tau_12, tau_13, tau_22, tau_23, 
                                                       tau_33, uf, vf, wf, tke, theta, #grad,
                                                                          train_pct = train_pct, size = size,
                                                                         augmentation = augmentation)
    
    print(x_train.shape)
    print(x_mu)
    print(x_std)
    
    # Train the Model
    models_final = [model_func(x_train, x_valid, x_test, tau_11_train, tau_11_valid, tau_11_test, 
                               act_func = 'relu', size = size, xdim = xdim),
                    model_func(x_train, x_valid, x_test, tau_12_train, tau_12_valid, tau_12_test, 
                               act_func = 'relu', size = size, xdim = xdim),
                    model_func(x_train, x_valid, x_test, tau_13_train, tau_13_valid, tau_13_test, 
                               act_func = 'relu', size = size, xdim = xdim),
                    model_func(x_train, x_valid, x_test, tau_22_train, tau_22_valid, tau_22_test, 
                               act_func = 'relu', size = size, xdim = xdim),
                    model_func(x_train, x_valid, x_test, tau_23_train, tau_23_valid, tau_23_test, 
                               act_func = 'relu', size = size, xdim = xdim),
                    model_func(x_train, x_valid, x_test, tau_33_train, tau_33_valid, tau_33_test, 
                               act_func = 'relu', size = size, xdim = xdim)]
    
    # Visualize Results
    results = [predict(models_final[0], x_test[...,:xdim], tau_11_test, mu_11, std_11),
               predict(models_final[1], x_test[...,:xdim], tau_12_test, mu_12, std_12),
               predict(models_final[2], x_test[...,:xdim], tau_13_test, mu_13, std_13),
               predict(models_final[3], x_test[...,:xdim], tau_22_test, mu_22, std_22),
               predict(models_final[4], x_test[...,:xdim], tau_23_test, mu_23, std_23),
               predict(models_final[5], x_test[...,:xdim], tau_33_test, mu_33, std_33)]
    
    visualize(models_final[0], x_test[...,:xdim], tau_11_test, mu_11, std_11)
    visualize(models_final[1], x_test[...,:xdim], tau_12_test, mu_12, std_12)
    visualize(models_final[2], x_test[...,:xdim], tau_13_test, mu_13, std_13)
    visualize(models_final[3], x_test[...,:xdim], tau_22_test, mu_22, std_22)
    visualize(models_final[4], x_test[...,:xdim], tau_23_test, mu_23, std_23)
    visualize(models_final[5], x_test[...,:xdim], tau_33_test, mu_33, std_33)
    
    return (models_final, results, mu_11, std_11, mu_12, std_12, mu_13, std_13, 
            mu_22, std_22, mu_23, std_23, mu_33, std_33)
    


# In[82]:


def save_model(model, label, mu_std):
    model_yaml = model.to_yaml()
    with open(label + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(label + '.h5')
    print("Saved model to disk")
    with open(label, 'wb') as fp:
        pickle.dump(mu_std, fp)


# In[83]:


def load_model(label):
    yaml_file = open(label + '.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(label + ".h5")
    print("Loaded model from disk")  
    with open(label, 'rb') as fp:
        mu_std = pickle.load(fp)
        
    return loaded_model, mu_std


# In[84]:


def run_model(test_dir, num_inputs = 3, model_exist = False):
    
    model_names = ['tau_11_test', 'tau_12_test', 'tau_13_test', 'tau_22_test', 'tau_23_test', 'tau_33_test']
    
    if not model_exist:
        prefix = 'Re15sh20_DNScoarse8/'

        model_conv_3d_feng, results_conv_3d_feng, mu_11, std_11, mu_12, std_12, mu_13, std_13, mu_22, std_22, mu_23, std_23, mu_33, std_33 = main_feng('conv-3d NN Feng', 3, None, num_inputs, prefix)

        mu_std = [(mu_11, std_11), (mu_12, std_12), (mu_13, std_13), 
                  (mu_22, std_22), (mu_23, std_23), (mu_33, std_33)]

        for x,y,z in zip(model_conv_3d_feng, model_names, mu_std):
            save_model(x, y, z)
            
    else:
        model_conv_3d_feng, mu_std = zip(*[load_model(x) for x in model_names])
        mu_all, std_all = zip(*mu_std)
        mu_11, mu_12, mu_13, mu_22, mu_23, mu_33 = mu_all
        std_11, std_12, std_13, std_22, std_23, std_33 = std_all
    
    # Output Data - CHANGE DATASET NAME HERE
    model_name, size, augmentation, xdim,  prefix = 'conv-3d NN Feng Test', 3, None, num_inputs, test_dir
    
    tau_11, tau_12, tau_13, tau_22, tau_23, tau_33 = get_output_data(prefix)
    print('Shape of Output Files:')
    print(tau_11.shape, tau_12.shape, tau_13.shape, tau_22.shape, tau_23.shape, tau_33.shape)
    
    # Input Data
    uf, vf, wf, tke, theta, grad = get_input_data(prefix)
    print('Shape of Input Files:')
    print(wf.shape)
        
    # Get Functions
    train_test_split_func, model_func = models[model_name]
    
    # Reshape Data and Get Train/Test Sets
    (x_train, _, x_test, tau_11_train, _, tau_11_test, 
     tau_12_train, _, tau_12_test, tau_13_train, _, tau_13_test,
     tau_22_train, _, tau_22_test, tau_23_train, _, tau_23_test, tau_33_train, _, tau_33_test,
     _, _, _, _, _, _, 
     _, _, _, _, _, _,_,_) = train_test_split_func(tau_11, tau_12, tau_13, tau_22, tau_23, 
                                                       tau_33, uf, vf, wf, tke, theta, #grad,
                                                                          train_pct = 0.5, size = size,
                                                                         augmentation = augmentation)
    
    y_test, model, mu, std = tau_11_test, model_conv_3d_feng[0], mu_11, std_11

    y_test = y_test.flatten()
    tau_11_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_11_predict))

    y_test, model, mu, std = tau_12_test, model_conv_3d_feng[1], mu_12, std_12

    y_test = y_test.flatten()
    tau_12_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_12_predict))

    y_test, model, mu, std = tau_13_test, model_conv_3d_feng[2], mu_13, std_13

    y_test = y_test.flatten()
    tau_13_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_13_predict))

    y_test, model, mu, std = tau_22_test, model_conv_3d_feng[3], mu_22, std_22

    y_test = y_test.flatten()
    tau_22_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_22_predict))

    y_test, model, mu, std = tau_23_test, model_conv_3d_feng[4], mu_23, std_23

    y_test = y_test.flatten()
    tau_23_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_23_predict))

    y_test, model, mu, std = tau_33_test, model_conv_3d_feng[5], mu_33, std_33

    y_test = y_test.flatten()
    tau_33_predict = model.predict(x_test[...,:xdim]).flatten()*std + mu

    print(np.corrcoef(y_test, tau_33_predict))
    
    return tau_11_predict, tau_12_predict, tau_13_predict, tau_22_predict, tau_23_predict, tau_33_predict, uf


# In[85]:


def main(test_dir, num_inputs = 3, model_exist = False):
    tau_11_predict, tau_12_predict, tau_13_predict, tau_22_predict, tau_23_predict, tau_33_predict, uf = run_model(test_dir, num_inputs = num_inputs, model_exist = model_exist)
    scipy.io.savemat(test_dir + 'tau_11_predict_test.mat', 
                 mdict={'tau_11': tau_11_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})
    scipy.io.savemat(test_dir + 'tau_12_predict_test.mat', 
                 mdict={'tau_12': tau_12_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})
    scipy.io.savemat(test_dir + 'tau_13_predict_test.mat', 
                 mdict={'tau_13': tau_13_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})
    scipy.io.savemat(test_dir + 'tau_22_predict_test.mat', 
                 mdict={'tau_22': tau_22_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})
    scipy.io.savemat(test_dir + 'tau_23_predict_test.mat', 
                 mdict={'tau_23': tau_23_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})
    scipy.io.savemat(test_dir + 'tau_33_predict_test.mat', 
                 mdict={'tau_33': tau_33_predict.reshape(uf.shape[0]-2,uf.shape[1]-2,int((uf.shape[2]*3/4)-1))})


# In[74]:


main('Re15sh20_DNScoarse16/', num_inputs = 3, model_exist = False)

