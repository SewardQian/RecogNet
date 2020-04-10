import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision.models

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/2020\ Winter/APS360/Project

vgg_m = torchvision.models.vgg16_bn(True).features.to(device).eval()
for param in vgg_m.parameters():
    param.requires_grad = False

blocks = [i-1 for i, o in enumerate(vgg_m.children()) if isinstance(o, nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]

def _hook_fn(m, i, o):
        return o if isinstance(o, torch.Tensor) else o if isinstance(o,(list, tuple)) else list(o)

_base_loss = F.l1_loss

def _gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

class Hook():
    def __init__(self, m, hook_func, detach=False):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if self.detach:
            input  = (o.detach() for o in input) if isinstance(input,  (list, tuple)) else input.detach()
            output = (o.detach() for o in input) if isinstance(output, (list, tuple)) else output.detach()
        self.stored = self.hook_func(module, input, output)

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = [Hook(m, _hook_fn, detach=False) for m in self.loss_features]
        self.wgts = layer_wgts
        self.matric_names = ["pixel", ] + [f"feat_{i}" for i in range(len(layer_ids))] + \
                            [f"gram_{i}" for i in range(len(layer_ids))]

    

    def make_features(self, x, clone=False):
        self.m_feat(x) 
        return [(o.stored.clone() if clone else o.stored) for o in self.hooks]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [_base_loss(input, target)]
        self.feat_losses += [_base_loss(f_in, f_out)*w 
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [_base_loss(_gram_matrix(f_in), _gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.matric_names, self.feat_losses))
        return sum(self.feat_losses)

import os
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

class UTKFaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.path, self.dirs, self.files = os.walk(root_dir).__next__()
        self.files = self.files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        image = io.imread(img_name)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

################################################stag1###################################
class Stg1En(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 16)
        self.down2 = Down(16, 16)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        return x

class Stg1De(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super().__init__()
        self.up1 = Up(16, 16, bilinear)
        self.up2 = Up(16, 16, bilinear)
        self.outc = OutConv(16, n_channels)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.outc(x)
        return x
class Stg1AE(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super().__init__()
        self.name = "Stg1AE"
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.En = Stg1En(self.n_channels)
        self.De = Stg1De(self.n_channels)
        

    def forward(self, x, debug=False):
        x = self.En(x)
        if debug: print(x.shape)
        x = self.De(x)
        if debug: print(x.shape)
        return x

from torch.utils.data import DataLoader

def train(model, train_loader, criterion, num_epochs=5, learning_rate=1e-4, chkpt_path=None, vis_path=None, starting_iter=0):
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter_n = starting_iter

    if iter_n == 0:
        if vis_path:
            visualize(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
        
        if chkpt_path:
            torch.save(model, chkpt_path + "_" + str(iter_n))

    for epoch in range(num_epochs):
        for data in train_loader:
            data = data.float().to(device)
            recon = model(data.clone())
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_n += 1
            print("Iter: {:5d} Loss: {:2.4f}".format(iter_n, float(loss)))

            if iter_n % 1000 == 0:
                if vis_path:
                    visualize(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
                
                if chkpt_path:
                    torch.save(model, chkpt_path + "_" + str(iter_n))
    
    if vis_path:
        visualize(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
    
    if chkpt_path:
        torch.save(model, chkpt_path + "_" + str(iter_n))

import matplotlib.pyplot as plt
import numpy as np

def visualize(model, dataset, show=True, path=None):

    assert len(dataset) >= 10, "Dataset Not Large Enough"

    fig = plt.figure(figsize=(7, 10))
    indices = np.random.choice(list(range(len(dataset))), size=10)

    for idx, id in enumerate(indices, 1):
        ax = fig.add_subplot(5, 4, idx * 2 - 1, xticks=[], yticks=[])
        plt.imshow(dataset[id].permute(1, 2, 0))
        ax = fig.add_subplot(5, 4, idx * 2, xticks=[], yticks=[])
        output = model(dataset[id].unsqueeze(0).to(device)).detach().squeeze().permute(1, 2, 0).cpu().numpy().clip(0., 1.)
        plt.imshow(output)
        # ax.set_title(idx)
    
    if show:
        plt.show()
    if path:
        fig.savefig(path)

#precompute stage1 feature to train stg2
stg1_dataset = h5py.File("/content/drive/My Drive/Project/faceScrub/img_stg_feat.h5", "r+")
for id in range(1000):

    data=hf["train_img"][id,...]
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = Image.fromarray(data)
    data = transform(data)
    data = torch.Tensor(data)
    data = data.to(device)

    out = stg1_en(data.unsqueeze(0))

    stg1_feature[id] = out.cpu().detach().squeeze().numpy()
hf.close()
stg1_dataset.close()

#load stg1 encoder decoder
test_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg1AE_1100")

test_m.eval()

for param in test_m.parameters():
    param.requires_grad = False

stg1_en = test_m.En

stg1_de = test_m.De

#########################################stage 2####################################
import os
from torch.utils.data import Dataset
import h5py

#construct dataset
class Stage2Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, lbl_path, feat_path, transform=None):
        super().__init__()
        self.transform = transform
        self.lbl_file = h5py.File(lbl_path, "r")
        self.feat_file = h5py.File(feat_path, "r+")
        self.labels = self.lbl_file["train_img"]
        self.features = self.feat_file["stg1_feature"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label= self.labels[idx]

        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = Image.fromarray(label)

        feature = self.features[idx]

        if self.transform:
            # feature = self.transform(feature)
            label = self.transform(label)
        
        feature = torch.Tensor(feature)
        label = torch.Tensor(label)
        # label=label.permute(2,0,1)

        return feature, label

stg2_transform = transforms.Compose([transforms.ToTensor()])
stg2_dataset = Stage2Dataset("/content/drive/My Drive/Project/faceScrub/1000_img.h5", "/content/drive/My Drive/Project/faceScrub/img_stg_feat.h5", transform=stg2_transform)
stg2_loader = DataLoader(dataset=stg2_dataset, batch_size=32, num_workers=8, shuffle=True)

class Stg2En(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        return x

class Stg2De(nn.Module):
    def __init__(self, n_channels=16, bilinear=True):
        super().__init__()
        self.up1 = Up(64, 64, bilinear)
        self.up2 = Up(64, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.outc(x)
        return x

class Stg2AE(nn.Module):
    def __init__(self, n_channels=16, bilinear=True):
        super().__init__()
        self.name = "Stg2AE"
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.En = Stg2En(self.n_channels)
        self.De = Stg2De(self.n_channels)
        

    def forward(self, x, debug=False):
        x = self.En(x)
        if debug: print(x.shape)
        x = self.De(x)
        if debug: print(x.shape)
        return x

stg2_m = Stg2AE()
stg2_m.to(device)
print(stg2_dataset[11][0])
stg2_m(stg2_dataset[11][0].unsqueeze(0).to(device), debug=True)

def visualize_stg2(model, dataset, show=True, path=None):

    assert len(dataset) >= 10, "Dataset Not Large Enough"

    fig = plt.figure(figsize=(7, 10))
    indices = np.random.choice(list(range(len(dataset))), size=10)

    for idx, id in enumerate(indices, 1):
        ax = fig.add_subplot(5, 4, idx * 2 - 1, xticks=[], yticks=[])
        print(dataset[id][1].permute(1,2,0).shape)
        # print(dataset[id][0].permute(2,0,1).shape)
        plt.imshow(dataset[id][1].permute(1,2,0))#.permute(1, 2, 0))
        ax = fig.add_subplot(5, 4, idx * 2, xticks=[], yticks=[])
        output = stg1_de(model(dataset[id][0].unsqueeze(0).to(device))).detach().squeeze().permute(1, 2, 0).cpu().numpy().clip(0., 1.)
        plt.imshow(output)
        # ax.set_title(idx)
    
    if show:
        plt.show()
    if path:
        fig.savefig(path)

from torch.utils.data import DataLoader
from IPython.display import clear_output

def train_stg2(model, train_loader, criterion, num_epochs=10, learning_rate=1e-4, chkpt_path=None, vis_path=None, starting_iter=0):
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter_n = starting_iter

    if iter_n == 0:
        if vis_path:
            visualize_stg2(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
        
        if chkpt_path:
            torch.save(model, chkpt_path + "_" + str(iter_n))

    for epoch in range(num_epochs):
        for data,label in train_loader:
            label = label.float().to(device)
            data= data.to(device)
            # print(data.shape,label.shape)
            # data = stg1_en(data)
      
            recon = model(data)
            recon_de = stg1_de(recon)
            # print(recon_de.shape,label.shape)
            loss = criterion(recon_de, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_n += 1
            print("Iter: {:5d} Loss: {:2.4f}".format(iter_n, float(loss)))

            if iter_n % 100 == 0:
                
                
                if vis_path:
                    visualize_stg2(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
                
                if chkpt_path:
                    torch.save(model, chkpt_path + "_" + str(iter_n))
    
    if vis_path:
        visualize_stg2(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
    
    if chkpt_path:
        torch.save(model, chkpt_path + "_" + str(iter_n))

#train stg2
output_clear = clear_output
train_stg2(stg2_m, stg2_loader, feat_loss, 100, 1e-3, "/content/drive/My Drive/Project/checkpoint2/" + stg2_m.name, "./visualization2/" + stg2_m.name, starting_iter=2100)

#load stg2 encoder decoder

stg2_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg2AE_5100")
stg2_m.eval()

for param in stg2_m.parameters():
    param.requires_grad = False

stg2_en = stg2_m.En
stg2_de = stg2_m.De

#precompute stg2 feature
transform= transforms.Compose([transforms.ToTensor()])
hf = h5py.File("/content/drive/My Drive/Project/faceScrub/img.h5", "r")
stg1_dataset = h5py.File("/content/drive/My Drive/Project/faceScrub/img_stg2_feat2.h5", "r+")

for id in range(1000):
    data=hf['train_img'][id,...]
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = Image.fromarray(data)
    
    data = transform(data)
    data = torch.Tensor(data)
    data = data.to(device)

    out = stg2_en(stg1_en(data.unsqueeze(0)))

    stg1_dataset["stg2_feature"][id] = out.cpu().detach().squeeze().numpy()

hf.close()
stg1_dataset.close()


#############################################stage 3 ###############################
class Stg3En(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.linear = nn.Linear(64*7*7, 1000)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = x.view((-1, 64*7*7))
        x = self.linear(x)
        return x

class Stg3De(nn.Module):
    def __init__(self, n_channels=64, bilinear=True):
        super().__init__()
        self.linear = nn.Linear(1000, 64*7*7)
        self.up1 = Up(64, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x = self.linear(x)
        x = x.view((-1, 64, 7, 7))
        x = self.up1(x)
        x = self.outc(x)
        return x

class Stg3AE(nn.Module):
    def __init__(self, n_channels=64, bilinear=True):
        super().__init__()
        self.name = "Stg3AE"
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.En = Stg3En(self.n_channels)
        self.De = Stg3De(self.n_channels)
        

    def forward(self, x, debug=False):
        x = self.En(x)
        if debug: print(x.shape)
        x = self.De(x)
        if debug: print(x.shape)
        return x

import os
from torch.utils.data import Dataset
import h5py


class Stage3Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, lbl_path, feat_path, transform=None):
        super().__init__()
        self.transform = transform
        self.lbl_file = h5py.File(lbl_path, "r")
        self.feat_file = h5py.File(feat_path, "r+")
        self.labels = self.lbl_file["train_img"]
        self.features = self.feat_file["stg2_feature"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label= self.labels[idx]

        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = Image.fromarray(label)

        feature = self.features[idx]

        if self.transform:
            # feature = self.transform(feature)
            label = self.transform(label)
        
        feature = torch.Tensor(feature)
        label = torch.Tensor(label)
        # label=label.permute(2,0,1)

        return feature, label

stg2_transform = transforms.Compose([transforms.ToTensor()])
stg3_dataset = Stage3Dataset("/content/drive/My Drive/Project/faceScrub/img.h5", "/content/drive/My Drive/Project/faceScrub/img_stg2_feat2.h5", transform=stg2_transform)
stg3_loader = DataLoader(dataset=stg3_dataset, batch_size=32, num_workers=8, shuffle=True)

def visualize_stg3(model, dataset, show=True, path=None):

    assert len(dataset) >= 10, "Dataset Not Large Enough"

    fig = plt.figure(figsize=(7, 10))
    indices = np.random.choice(list(range(len(dataset))), size=10)

    for idx, id in enumerate(indices, 1):
        ax = fig.add_subplot(5, 4, idx * 2 - 1, xticks=[], yticks=[])
        # print(dataset[id][1].permute(1,2,0).shape)
        # print(dataset[id][0].permute(2,0,1).shape)
        plt.imshow(dataset[id][1].permute(1,2,0))#.permute(1, 2, 0))
        ax = fig.add_subplot(5, 4, idx * 2, xticks=[], yticks=[])
        output = stg1_de(stg2_de(model(dataset[id][0].unsqueeze(0).to(device)))).detach().squeeze().permute(1, 2, 0).cpu().numpy().clip(0., 1.)
        # output = stg1_de(stg2_de((dataset[id][0].unsqueeze(0).to(device)))).detach().squeeze().permute(1, 2, 0).cpu().numpy().clip(0., 1.)
        plt.imshow(output)
        # ax.set_title(idx)
    
    if show:
        plt.show()
    if path:
        fig.savefig(path)

from torch.utils.data import DataLoader
from IPython.display import clear_output

def train_stg3(model, train_loader, criterion, num_epochs=5, learning_rate=1e-4, chkpt_path=None, vis_path=None, starting_iter=0):
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter_n = starting_iter

    if iter_n == 0:
        if vis_path:
            visualize_stg3(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
        
        if chkpt_path:
            torch.save(model, chkpt_path + "_" + str(iter_n))

    for epoch in range(num_epochs):
        for data, label in train_loader:
            label = label.float().to(device)
            data = data.float().to(device)
            # data = label.clone()
            # data = stg1_en(data)
            # data = stg2_en(data)
            recon = model(data)
            recon_de = stg2_de(recon)
            recon_de = stg1_de(recon_de)
            loss = criterion(recon_de, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_n += 1
            print("Iter: {:5d} Loss: {:2.4f}".format(iter_n, float(loss)))

            if iter_n % 100 == 0:
          
                
                if vis_path:
                    visualize_stg3(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
                
                if chkpt_path:
                    torch.save(model, chkpt_path + "_" + str(iter_n))
    
    if vis_path:
        visualize_stg3(model, train_loader.dataset, show=False, path=vis_path + "_" + str(iter_n))
    
    if chkpt_path:
        torch.save(model, chkpt_path + "_" + str(iter_n))

#train stage 3 
stg3_m = Stg3AE()
stg3_m.to(device)
train_stg3(stg3_m, stg3_loader, feat_loss, 200, 3e-3, "/content/drive/My Drive/Project/checkpoint2/" + stg3_m.name, "./visualization2/" + stg3_m.name, starting_iter=0)

#load stage 3 encoder decoder
stg3_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg3AE_6400")

stg3_m.eval()

for param in stg3_m.parameters():
    param.requires_grad = False

stg3_en = stg3_m.En



stg3_de = stg3_m.De