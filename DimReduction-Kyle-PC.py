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

#%%-------------------------old data loading-------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mat6=spio.loadmat(r"R:\SpinwAI\Training\Matrix6.mat")['Matrix']
J6=spio.loadmat(r"R:\SpinwAI\Training\J6.mat")['J']

mat4=spio.loadmat(r"R:\SpinwAI\Training\Matrix4.mat")['Matrix']
J4=spio.loadmat(r"R:\SpinwAI\Training\J4.mat")['J']

mat5_1=spio.loadmat(r"R:\SpinwAI\Training\Matrix5_1.mat")['Matrix5_1']
J5_1=spio.loadmat(r"R:\SpinwAI\Training\J5_1.mat")['J5_1']

mat5_2=spio.loadmat(r"R:\SpinwAI\Training\Matrix5_2.mat")['Matrix5_2']
J5_2=spio.loadmat(r"R:\SpinwAI\Training\J5_2.mat")['J5_2']

mat7=spio.loadmat(r"R:\SpinwAI\Training\Matrix7.mat")['Matrix']
J7=spio.loadmat(r"R:\SpinwAI\Training\J7.mat")['J']


mat=np.array([])
J=np.array([])
mat=np.concatenate((mat4,mat6,mat5_1,mat5_2, mat7),axis=2)
J=np.concatenate((J4,J6,J5_1,J5_2,J7),axis=0)
del J4, J6, J5_1, J5_2, J7
del mat4, mat6, mat5_1,mat5_2,mat7
#%%-------------------Data Prep-----------------
#plt.imshow(mat[:,:,1],extent=[0,500,0,500])
matT=mat.transpose((2,0,1))
mat_reshape=np.reshape(matT,(5000,249500))
data=mat_reshape
J_train, J_test, data_train, data_test=train_test_split(J,data,test_size=0.2)
J_train=torch.from_numpy(J_train).to(torch.float32).to(device)
J_test=torch.from_numpy(J_test).to(torch.float32).to(device)
data_train=torch.from_numpy(data_train).to(torch.float32).to(device)
data_test=torch.from_numpy(data_test).to(torch.float32).to(device)
J_train=torch.from_numpy(J_train)
J_test=torch.from_numpy(J_test)
data_train=torch.from_numpy(data_train)
data_test=torch.from_numpy(data_test)

matT=torch.from_numpy(matT).to(torch.float32).to(device)
J=torch.from_numpy(J).to(torch.float32)
J_train=J_train.to(torch.float32)
J_test=J_test.to(torch.float32)
data_train=data_train.to(torch.float32)
data_test=data_test.to(torch.float32)

#%%-----------------Dimensional Reduction
mat6=spio.loadmat(r"R:\SpinwAI\Training\Matrix6.mat")['Matrix']
J6=spio.loadmat(r"R:\SpinwAI\Training\J6.mat")['J']

mat4=spio.loadmat(r"R:\SpinwAI\Training\Matrix4.mat")['Matrix']
J4=spio.loadmat(r"R:\SpinwAI\Training\J4.mat")['J']

mat5_1=spio.loadmat(r"R:\SpinwAI\Training\Matrix5_1.mat")['Matrix5_1']
J5_1=spio.loadmat(r"R:\SpinwAI\Training\J5_1.mat")['J5_1']

mat5_2=spio.loadmat(r"R:\SpinwAI\Training\Matrix5_2.mat")['Matrix5_2']
J5_2=spio.loadmat(r"R:\SpinwAI\Training\J5_2.mat")['J5_2']

mat7=spio.loadmat(r"R:\SpinwAI\Training\Matrix7.mat")['Matrix']
J7=spio.loadmat(r"R:\SpinwAI\Training\J7.mat")['J']

mat=np.array([])
J=np.array([])
mat=np.concatenate((mat4,mat6,mat5_1,mat5_2, mat7),axis=2)
J=np.concatenate((J4,J6,J5_1,J5_2,J7),axis=0)
del J4, J6, J5_1, J5_2, J7
del mat4, mat6, mat5_1,mat5_2,mat7
matT=mat.transpose((2,0,1))
mat_reshape=np.reshape(matT,(5000,249500))

pca=PCA(n_components=32)
data=pca.fit_transform(mat_reshape)

data_back=pca.inverse_transform(data)
data_back_reshape=np.reshape(data_back,(5000,499,500))
plt.imshow(data_back_reshape[0,:,:], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)

for i in range(0,10):
    fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(10,8))

    ax0.imshow(mat[:,:,i], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)
    ax0.set_title('Original')

    ax1.imshow(data_back_reshape[i,:,:], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)
    ax1.set_title('Reconstructed')

    diff=ax2.imshow(mat[:,:,i]-data_back_reshape[i,:,:], extent=[0,500,0,500],cmap='jet',vmin=-1,vmax=1)
    fig.colorbar(diff,ax=ax2,shrink=0.3)

#%%----------Neural Network--------------------------------
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        # self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Linear function  # LINEAR
        x = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        x = torch.relu(x)

        x = self.fc2(x)

        x = torch.relu(x)

        # Linear function (readout)  # LINEAR
        x = self.fcout(x)
        return x

input_dim = 12
hidden_dim = 200
output_dim = 100

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
epoch = 10000
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    data_pred = model(J_train)
    # Compute Loss
    loss = criterion(data_pred, data_train)

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()

output=model(J_test[0:1,:])
output=output.detach().cpu().numpy()
output_back=pca.inverse_transform(output)
output_back_reshape=np.reshape(output_back,(1,499,500))
plt.imshow(output_back_reshape[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')

compare=data_test[0:1,:].detach().cpu().numpy()
compare=pca.inverse_transform(compare)
compare=np.reshape(compare,(1,499,500))
plt.imshow(compare[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
#%%---------------CNN Implementation----------------------------
# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=50, out_channels=30, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(3062800, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)


        out = out.reshape(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

criterion = nn.MSELoss()

learning_rate = 0.01

model = ConvNeuralNet(matT)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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


#%%-------------Autoencoder-----------------------------------------
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vec_dim = 12
        self.img_dim = 102400
        self.hidden_dim = 200

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(kwargs["input_shape"], 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(128, 256),
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
        self.fcout = nn.Linear(self.hidden_dim, 128)

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
epochs=1000



# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=249500).to(device)
model.train()
# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    loss = 0
    # reset the gradients back to zero
    # PyTorch accumulates gradients on subsequent backward passes
    optimizer.zero_grad()

    # # compute reconstructions
    # outputs = model(data_train)

    # # compute training reconstruction loss
    # train_loss = criterion(outputs, data_train)

    # compute forward
    encoded_out, recon_out, f_out = model.forward(data_train, J_train)

    train_loss = model.custom_loss(encoded_out, recon_out, f_out, data_train)


    # compute accumulated gradients
    train_loss.backward()

    # perform parameter update based on current gradients
    optimizer.step()

    # add the mini-batch training loss to epoch loss
    loss += train_loss.item()

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

model.eval()
#%%---------testing AE----------
# i=1
for i in range(10,20):
    predicted=model.inference(J_train[i])
    predicted=predicted.detach().cpu().numpy()
    predicted=np.reshape(predicted,(1,499,500))
    # plt.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    # plt.title('Autoencoder')
    # plt.colorbar()
    # plt.show()

    test=np.reshape(data_train[i].cpu(),(1,499,500))
    # plt.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    # plt.title('original')
    # plt.colorbar()
    # plt.show()

    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(10,10))

    cb0=ax0.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    ax0.set_title('original')
    fig.colorbar(cb0,ax=ax0,shrink=0.4)

    cb1=ax1.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
    ax1.set_title('Autoencoder')

    fig.colorbar(cb1,ax=ax1,shrink=0.4)
#%%-------------Save Model-----------------------------
EPOCH = epochs
PATH = r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\model.pt"
LOSS = 0.407494
torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
#%%-------------Load Model---------------------------------
PATH = r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\model.pt"
model = AE(input_shape=249500).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
