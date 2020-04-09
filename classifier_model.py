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
import h5py
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
from torch.utils.data import DataLoader
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataloader
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
import h5py
import random

#########################dataloader##############################

class classifierDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.lbl_file = h5py.File(path, "r")
 
        self.labels = self.lbl_file["names"]
        self.datas = self.lbl_file["data"]

    def __len__(self):
        return 9880 #len(self.features)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            

        data =self.datas[idx] 
        label = self.labels[idx] 

        label = int(label)

        return data, label

stg1_dataset = h5py.File("/content/drive/My Drive/Project/faceScrub/classifier_input2.h5", "r+")
stg2_transform = transforms.Compose([transforms.ToTensor()])
cdataset = classifierDataset("/content/drive/My Drive/Project/faceScrub/classifier_input2.h5",  transform=stg2_transform)
stg1_dataset.close()

max_v=int(16000)
indices_val = list(range(max_v))

random.shuffle(indices_val)

val_indices = indices_val[0:int(max_v*0.2)]
train_indices =indices_val[int(max_v*0.2):int(max_v*0.8)] 
test_indices = indices_val[:max_v]
  
  # Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices) #Samples elements randomly from
valid_sampler = SubsetRandomSampler(val_indices)   #the list of indices, without replacement.
test_sampler = SubsetRandomSampler(test_indices)

cloader_val = DataLoader(dataset=cdataset, batch_size=32,sampler=valid_sampler)
cloader_train = DataLoader(dataset=cdataset, batch_size=32,sampler=train_sampler)
cloader_test = DataLoader(dataset=cdataset, batch_size=32,sampler=test_sampler)

####################################classifier model ##################################
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.name = "classifier"
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, 30)
        self.fc4 = nn.Linear(30, 1)  
    def forward(self, x):
        x = x.view(-1,2000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

############################################ helper function ###############################
def evaluate(net, loader, criterion):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels

        outputs_positive = (outputs > 0.0).squeeze().long() == 1
        total_outputs_positive += int(outputs_positive.sum())

        outputs_positive_true = ((outputs > 0.0).squeeze().long() == 1) and (labels==1) 
        total_outputs_positive_true += int(outputs_positive.sum())

        output_neg_false = ((outputs > 0.0).squeeze().long() == 0) and (labels==1) 
        total_output_neg_false += int(output_neg_false.sum())

        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    precision = total_outputs_positive_true/total_outputs_positive
    print("precision: ", precision)

    recall = total_outputs_positive_true/(total_outputs_positive_true+total_output_neg_false)
    print("recall: ", recall)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    path="/content/drive/My Drive/Project/checkpoint2/"+path
    return path

def normalize_label(labels):
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)/(max_val - min_val)
    return norm_labels

# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def train_net(net, batch_size=32, learning_rate=0.01, num_epochs=20):
    
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(),lr=learning_rate)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(cloader_train, 0):
            # Get the inputs
            inputs, labels = data
            labels=torch.tensor(labels)
            # print(inputs.shape)
            # print(labels.shape)
            labels = normalize_label(labels) # Convert labels to 0/1
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        if(1):#epoch%3==0):
            train_err[epoch] = float(total_train_err) / total_epoch
            train_loss[epoch] = float(total_train_loss) / (i+1)
            val_err[epoch], val_loss[epoch] = evaluate(net, cloader_val, criterion)
            print(("Epoch {}: Train err: {}, Train loss: {} |"+
                  "Validation err: {}, Validation loss: {}").format(
                      epoch + 1,
                      train_err[epoch],
                      train_loss[epoch],
                      val_err[epoch],
                      val_loss[epoch]))
            # Save the current model (checkpoint) to a file
            model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
            print(model_path)
            torch.save(net.state_dict(), model_path)
    print('Finished Training')

    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

################################### start training ############################
net = classifier()
train_net(net,num_epochs=400,learning_rate=0.001)

#test loader result
criterion = nn.BCEWithLogitsLoss()

net = classifier()
state=torch.load("/content/drive/My Drive/Project/checkpoint2/model_classifier_bs32_lr0.001_epoch21")
net.load_state_dict(state)

val_err, val_loss = evaluate(net, cloader_test, criterion)
print(val_err,val_loss)
