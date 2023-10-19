# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:57:45 2022

@author: qmc
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as spio
import pickle as pk
import random


#%%-----------------Dimensional Reduction
mat1=spio.loadmat(r"R:\SpinwAI\Training\reduced\Mat_ebin\matsmall1.mat")['Matrix']
J1=spio.loadmat(r"R:\SpinwAI\Training\reduced\J_ebin\Jsmall1.mat")['J']

mat2=spio.loadmat(r"R:\SpinwAI\Training\reduced\Mat_ebin\matsmall2.mat")['Matrix']
J2=spio.loadmat(r"R:\SpinwAI\Training\reduced\J_ebin\Jsmall2.mat")['J']

mat3=spio.loadmat(r"R:\SpinwAI\Training\reduced\Mat_ebin\matsmall3.mat")['Matrix']
J3=spio.loadmat(r"R:\SpinwAI\Training\reduced\J_ebin\Jsmall3.mat")['J']

mat=np.array([])
J=np.array([])
mat=np.concatenate((mat1,mat2,mat3),axis=2)
J=np.concatenate((J1, J2, J3),axis=0)
del J1, J2, J3
del mat1, mat2, mat3
matT=mat.transpose((2,0,1))
mat_reshape=np.reshape(matT,(3000,62250))
#%%
pca=PCA(n_components=128)
data=pca.fit_transform(mat_reshape)
pk.dump(pca, open("pca\pca_128_3000.pkl","wb"))

data_back=pca.inverse_transform(data)
data_back_reshape=np.reshape(data_back,(3000,249,250))
# plt.imshow(data_back_reshape[0,:,:], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)

for j in range(0,10):
    i=random.randint(0,3000)
    fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(10,8))

    ax0.imshow(mat[:,:,i], extent=[0,250,0,250],cmap='jet',vmin=0,vmax=10,origin='lower')
    ax0.set_title('Original')

    ax1.imshow(data_back_reshape[i,:,:], extent=[0,250,0,250],cmap='jet',vmin=0,vmax=10,origin='lower')
    ax1.set_title('Reconstructed')

    diff=ax2.imshow(mat[:,:,i]-data_back_reshape[i,:,:], extent=[0,250,0,250],cmap='jet',vmin=-1,vmax=1,origin='lower')
    fig.colorbar(diff,ax=ax2,shrink=0.3)
#%%-----------pca experimental data--------------------
import cv2

exp=np.loadtxt("exp data/yb2o3.txt")
exp=exp[0:70,:]
for i in range(0,70):
    for j in range(0,470):
        if exp[i,j]<0.005 or exp[i,j]>0.05:
            exp[i,j]=0

# plt.imshow(exp,cmap="jet",vmin=0,vmax=0.02,origin='lower')
res = cv2.resize(exp, dsize=(250,249), interpolation=cv2.INTER_CUBIC)
plt.imshow(res,cmap="jet",vmin=0,vmax=0.02,origin='lower')
plt.colorbar()
plt.show()
scale=100
res=res*scale

res_reshape=np.reshape(res,(1,62250))
# expca=pca.transform(res_reshape)
expca=pca_reload.transform(res_reshape)

expcaback=pca_reload.inverse_transform(expca)
expcaback=np.reshape(expcaback, (249,250))
# plt.imshow(expcaback,cmap='jet',extent=[0,250,0,250],vmin=0,vmax=0.02*scale,origin='lower')
# plt.show()

for i in range(0,249):
    for j in range(0,250):
        if expcaback[i,j]<0.001*scale or expcaback[i,j]>0.05*scale:
            expcaback[i,j]=0

for j in range(100,150):
    for i in range(0,40):
        expcaback[i,j]=0

for j in range(0,249):
    for i in range(150,249):
        expcaback[i,j]=0


plt.imshow(expcaback,cmap='jet',extent=[0,250,0,250],vmin=0,vmax=0.02*scale,origin='lower')
plt.colorbar()
plt.show()
#%%
# later reload the pickle file
pca_reload = pk.load(open("pca\pca_128_3000.pkl",'rb'))
data=np.load("pca\pca_components_128.npy")
data_back = pca_reload.inverse_transform(data)
data_back_reshape=np.reshape(data_back,(3000,249,250))
#%%----------Neural Network--------------------------------
# class FeedforwardNeuralNetModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(FeedforwardNeuralNetModel, self).__init__()
#         # Linear function
#         self.fc1 = nn.Linear(input_dim, hidden_dim)

#         # Non-linearity
#         # self.sigmoid = nn.Sigmoid()

#         # Linear function (readout)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fcout = nn.Linear(hidden_dim, output_dim)


#     def forward(self, x):
#         # Linear function  # LINEAR
#         x = self.fc1(x)

#         # Non-linearity  # NON-LINEAR
#         x = torch.relu(x)

#         x = self.fc2(x)

#         x = torch.relu(x)

#         # Linear function (readout)  # LINEAR
#         x = self.fcout(x)
#         return x

# input_dim = 12
# hidden_dim = 200
# output_dim = 100

# model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

# criterion = nn.MSELoss()

# learning_rate = 0.001

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model.train()
# epoch = 10000
# for epoch in range(epoch):
#     optimizer.zero_grad()
#     # Forward pass
#     data_pred = model(J_train)
#     # Compute Loss
#     loss = criterion(data_pred, data_train)

#     print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
#     # Backward pass
#     loss.backward()
#     optimizer.step()

# output=model(J_test[0:1,:])
# output=output.detach().cpu().numpy()
# output_back=pca.inverse_transform(output)
# output_back_reshape=np.reshape(output_back,(1,499,500))
# plt.imshow(output_back_reshape[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')

# compare=data_test[0:1,:].detach().cpu().numpy()
# compare=pca.inverse_transform(compare)
# compare=np.reshape(compare,(1,499,500))
# plt.imshow(compare[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
# #%%---------------CNN Implementation----------------------------
# # Creating a CNN class
# class ConvNeuralNet(nn.Module):
# 	#  Determine what layers and their order in CNN object
#     def __init__(self, num_classes):
#         super(ConvNeuralNet, self).__init__()
#         self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3)
#         self.conv_layer2 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3)
#         self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.conv_layer3 = nn.Conv2d(in_channels=50, out_channels=30, kernel_size=3)
#         self.conv_layer4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3)
#         self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.fc1 = nn.Linear(3062800, 128)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)

#     # Progresses data across layers
#     def forward(self, x):
#         out = self.conv_layer1(x)
#         out = self.conv_layer2(out)
#         out = self.max_pool1(out)

#         out = self.conv_layer3(out)
#         out = self.conv_layer4(out)
#         out = self.max_pool2(out)


#         out = out.reshape(out.size(0), -1)
#         # print(out.size())
#         out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         return out

# criterion = nn.MSELoss()

# learning_rate = 0.01

# model = ConvNeuralNet(matT)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# model.train()
# epoch = 100000
# for epoch in range(epoch):
#     optimizer.zero_grad()
#     # Forward pass
#     data_pred = model(matT_small)
#     # Compute Loss
#     loss = criterion(data_pred, data_train)

#     print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
#     # Backward pass
#     loss.backward()
#     optimizer.step()


