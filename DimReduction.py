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

mat=spio.loadmat(r"C:\Users\qmc\OneDrive\ONRL\Data\SpinwAI\Matrix4.mat")
J=spio.loadmat(r"C:\Users\qmc\OneDrive\ONRL\Data\SpinwAI\J4.mat")
mat=mat['Matrix']
J=J['J']
#%%-------------------Data Prep-----------------
#plt.imshow(mat[:,:,1],extent=[0,500,0,500])
matT=mat.transpose((2,0,1))
mat_reshape=np.reshape(matT,(1000,249500))
data=mat_reshape
J_train, J_test, data_train, data_test=train_test_split(J,data,test_size=0.2)
J_train=torch.from_numpy(J_train).to(torch.float32)
J_test=torch.from_numpy(J_test).to(torch.float32)
data_train=torch.from_numpy(data_train).to(torch.float32)
data_test=torch.from_numpy(data_test).to(torch.float32)
#%%-----------------Dimensional Reduction
pca=PCA(n_components=100)
data=pca.fit_transform(mat_reshape)
J_train, J_test, data_train, data_test=train_test_split(J,data,test_size=0.2)
J_train=torch.from_numpy(J_train)
J_test=torch.from_numpy(J_test)
data_train=torch.from_numpy(data_train)
data_test=torch.from_numpy(data_test)

matT=torch.from_numpy(matT).to(torch.float32)
J=torch.from_numpy(J).to(torch.float32)
J_train=J_train.to(torch.float32)
J_test=J_test.to(torch.float32)
data_train=data_train.to(torch.float32)
data_test=data_test.to(torch.float32)

data_back=pca.inverse_transform(data)
data_back_reshape=np.reshape(data_back,(1000,499,500))
plt.imshow(data_back_reshape[0,:,:], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)


# for i in range(0,1):
#     fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(10,8))
    
#     ax0.imshow(mat[:,:,i], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)
#     ax0.set_title('Original')
    
#     ax1.imshow(data_back_reshape[i,:,:], extent=[0,500,0,500],cmap='jet',vmin=0,vmax=10)
#     ax1.set_title('Reconstructed')
    
#     diff=ax2.imshow(mat[:,:,i]-data_back_reshape[i,:,:], extent=[0,500,0,500],cmap='jet',vmin=-1,vmax=1)
#     fig.colorbar(diff,ax=ax2,shrink=0.3)

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
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.encoder_output_layer = nn.Linear(
            in_features=258, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

epochs=1000

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=249500).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()
for epoch in range(epochs):
    loss = 0
    # reset the gradients back to zero
    # PyTorch accumulates gradients on subsequent backward passes
    optimizer.zero_grad()
    
    # compute reconstructions
    outputs = model(data_train)
    
    # compute training reconstruction loss
    train_loss = criterion(outputs, data_train)
    
    # compute accumulated gradients
    train_loss.backward()
    
    # perform parameter update based on current gradients
    optimizer.step()
    
    # add the mini-batch training loss to epoch loss
    loss += train_loss.item()
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

#%%---------testing AE----------
i=1
predicted=model(data_test[i])
predicted=predicted.detach().cpu().numpy()
predicted=np.reshape(predicted,(1,499,500))
plt.imshow(predicted[0,:,:],cmap='jet',vmin=0,vmax=5,origin='lower')
plt.show()

test=np.reshape(data_test[i],(1,499,500))
plt.imshow(test[0,:,:],cmap='jet',vmin=0,vmax=10,origin='lower')
plt.show()