# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:57:45 2022

@author: qmc
"""
import h5py
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
#-------------------------Using data loader------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class NeutronData(Dataset):
#     def __init__(self, matlab_matrix, J_matrix, root_dir, transform=None):
#         self.mat=np.loadtxt(matlab_matrix)
#         self.J=np.loadtxt(J_matrix)
#         self.root_dir=root_dir
#         self.transform=transform

#     def __len__(self):
#         return len(self.J)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image=self.mat[idx]
#         J_par=self.J[idx]

#         if self.transform:
#             # image=np.array(image)
#             # image= np.reshape(image,249500)
#             # J_par=np.array(J_par)
#             image=torch.from_numpy(image).to(torch.float32).to(device)
#             J_par=torch.from_numpy(J_par).to(torch.float32).to(device)
#         return image, J_par

class NeutronData(Dataset):
    def __init__(self, matlab_matrix,Mkey, J_matrix, Jkey, root_dir, transform=None):

        # self.mat=pd.read_csv(matlab_matrix)
        self.mat=spio.loadmat(matlab_matrix)[Mkey]
        # self.J=pd.read_csv(J_matrix)
        self.J=spio.loadmat(J_matrix)[Jkey]
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.J)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image=self.mat[:,:,idx]
        J_par=self.J[idx]

        if self.transform:
            image=np.array(image)
            image= np.reshape(image,249500)
            J_par=np.array(J_par)
            image=torch.from_numpy(image).to(torch.float32).to(device)
            J_par=torch.from_numpy(J_par).to(torch.float32).to(device)
        return image, J_par

all_data=[]

neutron_dataset4 = NeutronData(matlab_matrix="Training/Matrix4.mat",
                               Mkey='Matrix',
                              J_matrix="Training/J4.mat",
                              Jkey='J',
                              root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset4)
del neutron_dataset4


neutron_dataset5_1 = NeutronData(matlab_matrix=r"Training/Matrix5_1.mat",
                                 Mkey='Matrix5_1',
                                J_matrix=r"Training/J5_1.mat",
                                Jkey='J5_1',
                                root_dir="SpinwAI/",transform=True)

all_data.append(neutron_dataset5_1)
del neutron_dataset5_1

neutron_dataset5_2 = NeutronData(matlab_matrix=r"Training/Matrix5_2.mat",
                                 Mkey='Matrix5_2',
                                J_matrix=r"Training/J5_2.mat",
                                Jkey='J5_2',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset5_2)
del neutron_dataset5_2

neutron_dataset5_3 = NeutronData(matlab_matrix=r"Training/Matrix5_3.mat",
                                 Mkey='Matrix5_3',
                                J_matrix=r"Training/J5_3.mat",
                                Jkey='J5_3',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset5_3)
del neutron_dataset5_3


neutron_dataset5_4 = NeutronData(matlab_matrix=r"Training/Matrix5_4.mat",
                                 Mkey='Matrix5_4',
                                J_matrix=r"Training/J5_4.mat",
                                Jkey='J5_4',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset5_4)
del neutron_dataset5_4

neutron_dataset5_5 = NeutronData(matlab_matrix=r"Training/Matrix5_5.mat",
                                 Mkey='Matrix5_5',
                                J_matrix=r"Training/J5_5.mat",
                                Jkey='J5_5',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset5_5)
del neutron_dataset5_5

neutron_dataset6 = NeutronData(matlab_matrix=r"Training/Matrix6.mat",
                                 Mkey='Matrix',
                                J_matrix=r"Training/J6.mat",
                                Jkey='J',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset6)
del neutron_dataset6

neutron_dataset6 = NeutronData(matlab_matrix=r"Training/Matrix7.mat",
                                 Mkey='Matrix',
                                J_matrix=r"Training/J7.mat",
                                Jkey='J',
                                root_dir="SpinwAI/",transform=True)
all_data.append(neutron_dataset6)
del neutron_dataset6

neutron_dataset_concat=ConcatDataset(all_data)
del all_data
batchsize=20
dataloader=DataLoader(neutron_dataset_concat, batch_size=batchsize,shuffle=True, num_workers=0)
del neutron_dataset_concat

#-------------Autoencoder-----------------------------------------
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vec_dim = 12
        # self.img_dim = 102400
        self.hidden_dim = 200

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(kwargs["input_shape"], 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, kwargs["input_shape"]),
            torch.nn.ReLU()
        )

        self.fc1 = nn.Linear(self.vec_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fcout = nn.Linear(self.hidden_dim, 64)

    def forward(self, features, par_vec):

        x = torch.relu(self.fc1(par_vec))
        x = torch.relu(self.fc2(x))
        f = self.fcout(x)

        encoded = self.encoder(features) #the image as features reduced to 128 components
        reconstructed = self.decoder(encoded) #decoded images from the 128 components

        return encoded, reconstructed, f

    def custom_loss(self, encoded, reconstructed, f, features):
        # mean-squared error loss
        criterion = nn.MSELoss()
        # AE loss
        loss1 = criterion(reconstructed, features)

        # FC loss
        loss2 = criterion(f, encoded)

        return loss1 + 10.0 * loss2

    def inference(self, par_vec):

        x = torch.relu(self.fc1(par_vec))
        x = torch.relu(self.fc2(x))
        f = self.fcout(x)

        reconstructed = self.decoder(f)

        return reconstructed


#%%----------------------training the AE-----------------------
if  __name__=='__main__':
    epochs=5

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=249500).to(device)
    model.train()
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    for epoch in range(epochs):
        loss=0
        # for i,data in enumerate(dataloader):
        # i=0
        for data_train, J_train in dataloader:
            # data_train=torch.from_numpy(data_train).to(torch.float32).to(device)
            # J_train=torch.from_numpy(J_train).to(torch.float32).to(device)
            optimizer.zero_grad()
            encoded_out, recon_out, f_out = model.forward(data_train, J_train)

            train_loss = model.custom_loss(encoded_out, recon_out, f_out, data_train)


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
#%%---------testing AE----------
# i=1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mat=spio.loadmat("Training\Matrix4.mat")
# J=spio.loadmat("Training\J4.mat")
# mat=mat['Matrix']
# J=J['J']
# matT=mat.transpose((2,0,1))
# mat_reshape=np.reshape(matT,(1000,249500))
# data=mat_reshape
# J_train, J_test, data_train, data_test=train_test_split(J,data,test_size=0.2)
# J_train=torch.from_numpy(J_train).to(torch.float32).to(device)
# J_test=torch.from_numpy(J_test).to(torch.float32).to(device)
# data_train=torch.from_numpy(data_train).to(torch.float32).to(device)
# data_test=torch.from_numpy(data_test).to(torch.float32).to(device)
# for i in range(10,20):
#     predicted=model.inference(J_train[i])
#     predicted=predicted.detach().cpu().numpy()
#     predicted=np.reshape(predicted,(1,499,500))
#     # plt.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
#     # plt.title('Autoencoder')
#     # plt.colorbar()
#     # plt.show()

#     test=np.reshape(data_train[i].cpu(),(1,499,500))
#     # plt.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
#     # plt.title('original')
#     # plt.colorbar()
#     # plt.show()

#     fig,(ax0,ax1)=plt.subplots(1,2,figsize=(10,10))

#     cb0=ax0.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
#     ax0.set_title('original')
#     fig.colorbar(cb0,ax=ax0,shrink=0.4)

#     cb1=ax1.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
#     ax1.set_title('Autoencoder')

#     fig.colorbar(cb1,ax=ax1,shrink=0.4)

# %%-------------Save Model-----------------------------
EPOCH = epochs
PATH = r"R:\SpinwAI\model_dataloader.pt"
LOSS = loss
torch.save({
            'epoch': EPOCH+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
#%%-------------Load Model---------------------------------
PATH = r"R:\SpinwAI\model_dataloader.pt"
model = AE(input_shape=249500).to(device)
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
    new_epochs=pre_epochs+5
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    # model = AE(input_shape=249500).to(device)
    model.train()
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3


    for ep in range(pre_epochs,new_epochs):
        loss=0
        # for i,data in enumerate(dataloader):
        # i=0
        for data_train, J_train in dataloader:
            # data_train=torch.from_numpy(data_train).to(torch.float32).to(device)
            # J_train=torch.from_numpy(J_train).to(torch.float32).to(device)
            optimizer.zero_grad()
            encoded_out, recon_out, f_out = model.forward(data_train, J_train)

            train_loss = model.custom_loss(encoded_out, recon_out, f_out, data_train)


            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            # i=i+1
            # print(i)
        ep=epoch
            # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, new_epochs, loss))
