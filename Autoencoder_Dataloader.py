# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:57:45 2022

@author: qmc
"""
# import h5py
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import scipy.io as spio
import cv2


import data_loader as ld
#-------------------------Using data loader------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader=ld.dataloader

exp=np.loadtxt("exp data/yb2o3.txt")
exp=exp[0:70,:]
# for i in range(0,70):
#     for j in range(0,470):
#         if exp[i,j]<0.005 or exp[i,j]>0.05:
#             exp[i,j]=0


res = cv2.resize(exp, dsize=(250,249), interpolation=cv2.INTER_CUBIC)
res=np.reshape(res,62250)

scale=1000
# plt.imshow(exp,cmap="jet",vmin=0,vmax=0.02,origin='lower')
# plt.colorbar()
# plt.show()
res=torch.from_numpy(res*scale).to(torch.float32).to(device)
#-------------Autoencoder-----------------------------------------
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.comp=128
        self.vec_dim = 12
        # self.img_dim = 102400
        # self.hidden_dim = 200
        self.hidden_dim = 384

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(kwargs["input_shape"], 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.comp),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(self.comp, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, kwargs["input_shape"]),
            torch.nn.ReLU()
        )

        self.fcout = nn.Linear(self.comp,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim,self.vec_dim)

    def forward(self, features, features_exp):

        encoded = self.encoder(features) #the image as features reduced to 128 components
        x = torch.relu(self.fcout(encoded))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        f = self.fc1(x)

        reconstructed = self.decoder(encoded) #decoded images from the 128 components

        encoded_exp=self.encoder(features_exp)
        reconstructed_exp = self.decoder(encoded_exp)

        return encoded, reconstructed, f, reconstructed_exp

    def custom_loss(self, J, reconstructed, f, features, reconstructed_exp,res):
        # mean-squared error loss
        criterion = nn.MSELoss()
        # AE loss
        loss1 = criterion(reconstructed, features)

        # FC loss
        loss2 = criterion(f, J)


        loss3 = criterion(reconstructed_exp, res)
        return loss1 + 20.0 * loss2 + 10*loss3

    def inference(self, features):

        encoded = self.encoder(features) #the image as features reduced to 128 components
        x = torch.relu(self.fcout(encoded))
        x = torch.relu(self.fc2(x))
        f = self.fc1(x)

        reconstructed = self.decoder(encoded) #decoded images from the 128 components

        return reconstructed, f



#%%----------------------Initial training the AE-----------------------
if  __name__=='__main__':
    epochs=5000

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=62250).to(device)
    model.train()
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


    for epoch in range(epochs):
        loss=0
        # for i,data in enumerate(dataloader):
        # i=0
        for data_train, J_train in dataloader:
            # data_train=torch.from_numpy(data_train).to(torch.float32).to(device,non_blocking=True)
            # J_train=torch.from_numpy(J_train).to(torch.float32).to(device,non_blocking=True)
            # data_train=data_train.to(torch.float32).to(device,non_blocking=True)
            # J_train=J_train.to(torch.float32).to(device,non_blocking=True)
            optimizer.zero_grad()
            encoded_out, recon_out, J_out, reconstructed_exp = model.forward(data_train,res)

            train_loss = model.custom_loss(J_train, recon_out, J_out, data_train, reconstructed_exp,res)


            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            # i=i+1
            # print(i)
            # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


# model.eval()
#%%--------------load experimental data--------------
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import scipy.io as spio
import numpy as np
import cv2
import matplotlib.pyplot as plt

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp=np.loadtxt("exp data/yb2o3.txt")
exp=exp[0:70,:]
# for i in range(0,70):
#     for j in range(0,470):
#         if exp[i,j]<0.005 or exp[i,j]>0.05:
#             exp[i,j]=0

plt.imshow(exp,cmap="jet",vmin=0,vmax=0.02,origin='lower')
res = cv2.resize(exp, dsize=(250,249), interpolation=cv2.INTER_CUBIC)

scale=1000
res=res*scale

# res=expcaback
res_reshape=np.reshape(res,62250)

plt.imshow(res,cmap="jet",origin='lower')
plt.colorbar()
plt.show()

res_reshape=torch.from_numpy(res_reshape).to(torch.float32).to(device)
predicted,J_pred=model.inference(res_reshape)
predicted=predicted.detach().cpu().numpy()
predicted=np.reshape(predicted,(249,250))
plt.imshow(predicted,cmap='jet', origin="lower",vmin=0,vmax=0.02*scale)
plt.colorbar()
plt.show()
#%%---------testing AE----------
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mat=spio.loadmat(r"Training\reduced\Mat_ebin\matsmall1.mat")
# J=spio.loadmat(r"Training\reduced\J_ebin\Jsmall1.mat")

mat=spio.loadmat(r"Training\matscale14.mat")
J=spio.loadmat(r"Training\Jscale14.mat")

mat=mat['Matrix']
J=J['J']
matT=mat.transpose((2,0,1))
mat_reshape=np.reshape(matT,(1000,62250))

data=mat_reshape
J_train, J_test, data_train, data_test=train_test_split(J,data,test_size=0.2)
J_train=torch.from_numpy(J_train).to(torch.float32).to(device)
J_test=torch.from_numpy(J_test).to(torch.float32).to(device)
data_train=torch.from_numpy(data_train).to(torch.float32).to(device)
data_test=torch.from_numpy(data_test).to(torch.float32).to(device)
for j in range(0,5):
    i=random.randint(0,800)
    predicted,J_pred=model.inference(data_train[i,:])
    predicted=predicted.detach().cpu().numpy()
    predicted=np.reshape(predicted,(1,249,250))
    # plt.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    # plt.title('Autoencoder')
    # plt.colorbar()
    # plt.show()

    test=np.reshape(data_train[i].cpu(),(1,249,250))
    # plt.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    # plt.title('original')
    # plt.colorbar()
    # plt.show()

    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(10,10))

    cb0=ax0.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=20,origin='lower')
    # ax0.set_title(J_train[i,:].detach().cpu().numpy().round(3),fontsize=8)
    ax0.set_title(i,fontsize=8)
    fig.colorbar(cb0,ax=ax0,shrink=0.4)

    cb1=ax1.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=20,origin='lower')
    # ax1.set_title(J_pred.detach().cpu().numpy().round(3),fontsize=8)
    fig.colorbar(cb1,ax=ax1,shrink=0.4)
    print("---------------------------")
    print('i=',i)
    print("Original=",J_train[i,:].numpy(force=True).round(3))
    print("Predicted=",J_pred.numpy(force=True).round(3))
# %%-------------Save Model-----------------------------
EPOCH = epochs
# EPOCH = new_epochs
PATH = r"R:\SpinwAI\model_J_scale.pt"
LOSS = loss
torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
#%%-------------Load Model---------------------------------
PATH = r"R:\SpinwAI\model_J_scale.pt"
model = AE(input_shape=62250).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.train()
#%%----------------resume training-----------------
if  __name__=='__main__':
    pre_epochs=epoch
    new_epochs=pre_epochs+5000
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    # model = AE(input_shape=249500).to(device)
    model.train()
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3


    for epoch in range(pre_epochs,new_epochs):
        loss=0
        # for i,data in enumerate(dataloader):
        # i=0
        for data_train, J_train in dataloader:
            # data_train=torch.from_numpy(data_train).to(torch.float32).to(device,non_blocking=True)
            # J_train=torch.from_numpy(J_train).to(torch.float32).to(device,non_blocking=True)
            # data_train=data_train.to(torch.float32).to(device,non_blocking=True)
            # J_train=J_train.to(torch.float32).to(device,non_blocking=True)
            optimizer.zero_grad()
            encoded_out, recon_out, J_out, reconstructed_exp = model.forward(data_train,res)

            train_loss = model.custom_loss(J_train, recon_out, J_out, data_train, reconstructed_exp,res)


            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, new_epochs, loss))


#%%-------------Save Model-----------------------------
# EPOCH = epochs
EPOCH = new_epochs
PATH = r"R:\SpinwAI\model_J_scale.pt"
LOSS = loss
torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)