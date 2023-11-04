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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-------------------------Using data loader------------------------------

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
            image= np.reshape(image,62250)
            J_par=np.array(J_par)
            image=torch.from_numpy(image).to(torch.float32).to(device)
            J_par=torch.from_numpy(J_par).to(torch.float32).to(device)
        return image, J_par

Jpath=r'R:\SpinwAI\Training\reduced\Jscale'
Matpath=r'R:\SpinwAI\Training\reduced\Matscale'
Jdir = os.listdir(Jpath)
Matdir = os.listdir(Matpath)
all_data=[]
for i in range(len(Jdir)):
# for i in range(0,1):
    neutron_dataset = NeutronData(matlab_matrix=os.path.join(Matpath, Matdir[i]),
                                      Mkey='Matrix',
                                    J_matrix=os.path.join(Jpath, Jdir[i]),
                                    Jkey='J',
                                    root_dir="SpinwAI/",transform=True)
    all_data.append(neutron_dataset)


neutron_dataset_concat=ConcatDataset(all_data)
del all_data
batchsize=1000
dataloader=DataLoader(neutron_dataset_concat, batch_size=batchsize,shuffle=True, num_workers=0)
del neutron_dataset_concat