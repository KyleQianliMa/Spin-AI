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
import random
import data_loader as ld
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------Autoencoder-----------------------------------------
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.comp=128
        self.vec_dim = 6
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

    def forward(self, features):

        encoded = self.encoder(features) #the image as features reduced to 128 components
        x = torch.relu(self.fcout(encoded))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        f = self.fc1(x)

        reconstructed = self.decoder(encoded) #decoded images from the 128 components


        return encoded, reconstructed, f

    def custom_loss(self, J, reconstructed, f, features):
        # mean-squared error loss
        criterion = nn.MSELoss()
        # AE loss
        loss1 = criterion(reconstructed, features)

        # FC loss
        loss2 = criterion(f, J)

        return 1.0*loss1 + 1.0 * loss2

    def inference(self, features):

        encoded = self.encoder(features) #the image as features reduced to 128 components
        x = torch.relu(self.fcout(encoded))
        x = torch.relu(self.fc2(x))
        f = self.fc1(x)

        reconstructed = self.decoder(encoded) #decoded images from the 128 components

        return reconstructed, f



#%%----------------------Initial training the AE-----------------------
def Training(epochs,model_name,dataloader,PATH):

    '''This function check if wether a model already exists in the PATH. If PATH exists it load and re-train and save model.
       If not it train a new model and save it.
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(input_shape=62250).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    #if model already exists, then we load previous model, re-train and save the model otherwise set pre-epochs=0 and start training
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        pre_epochs=epoch
        new_epochs=pre_epochs+epochs
        model.train()
    else:
        pre_epochs=0
        new_epochs=epochs

    for epoch in range(pre_epochs,new_epochs):
        loss=0

        for data_train, J_train in dataloader:

            optimizer.zero_grad()
            encoded_out, recon_out, J_out = model.forward(data_train)

            train_loss = model.custom_loss(J_train, recon_out, J_out, data_train)


            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, new_epochs, loss))
    EPOCH = new_epochs
    LOSS = loss
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, PATH)

    return model, loss, optimizer

def Testing(model_name,mat,J,count, PATH):
    '''This function load a model and print testing images

       model_name: pre-saved model name. model_name only  represents the material name
       mat_test: The path and name for testing data
       J_test:   The path and name for testing labels
       counts:   Number of random selected images user want to generate

       '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(input_shape=62250).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

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
    for j in range(0,count):
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

        cb0=ax0.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
        # ax0.set_title(J_train[i,:].detach().cpu().numpy().round(3),fontsize=8)
        ax0.set_title(i,fontsize=8)
        fig.colorbar(cb0,ax=ax0,shrink=0.4)

        cb1=ax1.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
        # ax1.set_title(J_pred.detach().cpu().numpy().round(3),fontsize=8)
        fig.colorbar(cb1,ax=ax1,shrink=0.4)
        print("---------------------------")
        print('i=',i)
        print("Original=",J_train[i,:].numpy(force=True).round(3))
        print("Predicted=",J_pred.numpy(force=True).round(3))

#%%---------testing AE----------
if  __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader=ld.dataloader
    model_name='Cu3TeO6.pt' #only contains the material name and .pt
    PATH = r"R:\SpinwAI\Cu3TeO6\model_"+model_name
    epochs=5 #change this number to around 1000 for initial training
    model,loss,optimizer=Training(epochs,model_name,dataloader,PATH)

    model_name='Cu3TeO6.pt' #change model name and path if we want to test different models
    PATH = r"R:\SpinwAI\Cu3TeO6\model_"+model_name
    mat_test=spio.loadmat(r"R:\SpinwAI\Cu3TeO6\Mat\mat01.mat") #01-19 sets of data have been seen. 20th data set is not seen
    J_test=spio.loadmat(r"R:\SpinwAI\Cu3TeO6\J\J01.mat") #01-19 sets of data have been seen. 20th data set is not seen
    count=2
    Testing(model_name,mat_test,J_test,count)
