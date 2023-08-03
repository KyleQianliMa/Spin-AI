import torch
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as spio
#%%----------
# testmat=spio.loadmat(r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\Matrix4.mat")['Matrix']
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NeutronData(Dataset):
    def __init__(self, matlab_matrix, J_matrix, root_dir, transform=None):
        try:
            self.mat=spio.loadmat(matlab_matrix)['Matrix']
        except:
            self.mat=h5py.File(matlab_matrix)['Matrix']
        self.J=spio.loadmat(J_matrix)['J']
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.J)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(self.mat) is h5py._hl.dataset.Dataset:
            image=self.mat[idx,:,:].transpose(1,0)
        else:
            image=self.mat[:,:,idx]
        J_par=self.J[idx]

        if self.transform:
            image= np.array(image)
            J_par=np.array(J_par)
            image=torch.from_numpy(image).to(torch.float32).to(device)
            J_par=torch.from_numpy(J_par).to(torch.float32).to(device)
        return image, J_par

neutron_dataset = NeutronData(matlab_matrix=r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\Matrix5.mat",
                              J_matrix=r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\J5.mat",
                              root_dir="SpinwAI/",transform=None)

dataloader=DataLoader(neutron_dataset, batch_size=4,shuffle=True, num_workers=0)

i=0
for features, labels in dataloader:
    print(labels)

# # feature,label=next(iter(dataloader))
# a=h5py.File(r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\Matrix5.mat")['Matrix']
# b=np.array(a)
# b=b.transpose(2,1,0)
# plt.imshow(b[:,:,0])
#%%--------test------------
matlab_matrix=r"C:\Users\kyleq\OneDrive\ONRL\Data\SpinwAI\Matrix5.mat"
try:
    mat=spio.loadmat(matlab_matrix)['Matrix']
except:
    mat=h5py.File(matlab_matrix)['Matrix']

if type(mat) is h5py._hl.dataset.Dataset:
    image=mat[0,:,:].transpose(1,0,2)
    image=np.reshape(image,(5000,249500))
img=mat[:,:,0]