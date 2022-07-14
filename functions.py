import torch
import numpy as np
import pandas as pd
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from random import randrange

import gc

import torch.nn as nn
import torch.nn.functional as F
from torch import nn

import time
def load_data_from_many_folders(rawDataset, labels, ProjectPath, folder_name, newLabel, label_to_name,
                                Print_Folders=False):
    # Get directories in AK folder
    r = None  # AK
    folders = None  # All directory names in AK
    f = None  # All files in AK

    i = 0
    path = ProjectPath + folder_name
    for root, dirs, files in os.walk(path, topdown=True):
        r = root
        folders = dirs
        f = files
        # print(f'root: {root}, dirs: {dirs}, files: {files}')
        break

    # Sort folders by name
    folders.sort()
    print(f'Number of folders in {folder_name}: {len(folders)}')

    transform = transforms.Compose([transforms.ToTensor()])

    folder_number = 0
    loaded_image_count = 0

    for folder in folders:  # Loop through each patietn to extract images
        pfiles = None
        # File path to folder
        path_to_folder = path + '/' + folder
        for root, dirs, files in os.walk(path_to_folder, topdown=True):
            count = 0  # Count number of images folder has
            pfiles = files

            image_names = []  # Store all image names in folder's folder

            for file in pfiles:
                if '.jpg' in file:
                    count += 1
                    image_names.append(file)

            for name in image_names:
                path_to_image = path_to_folder + '/' + name

                img = mpimg.imread(path_to_image)
                img_tensor = transform(img)  # normalizes the image also

                rawDataset.append(img_tensor)
                loaded_image_count += 1
                if loaded_image_count % 100 == 0:
                    print(f'Loaded {loaded_image_count} {label_to_name[newLabel]} images')

            if Print_Folders:
                print(f'Number {folder_number}: {folder} has {count} pictures')

            if count == 0:
                raise Exception(f'Error! folder {folder} has no pictures!. Please add pictures or remove folder.')
            break

        folder_number += 1

    newLabels = [newLabel for i in range(len(rawDataset))]

    labels.extend(newLabels)
    print(f'rawDataset now has a total of {len(rawDataset)} images')


# Load images from a single folder into rawDataset

def load_data_from_one_folder(rawDataset, labels, ProjectPath, folder_name, newLabel, label_to_name):
    f = None  # holds file names
    image_names = []  # hold image names
    transform = transforms.Compose([transforms.ToTensor()])
    path = ProjectPath + folder_name
    for root, dirs, files in os.walk(path, topdown=True):
        f = files
        break

    for name in f:
        if '.jpg' in name:
            image_names.append(name)

    count = 0  # number of FK images
    for i, name in enumerate(image_names):
        count += 1
        path_to_image = path + '/' + name
        img = mpimg.imread(path_to_image)

        img_tensor = transform(img)  # turn to tensor

        rawDataset.append(img_tensor)
        if (count) % 100 == 0 or count == len(image_names):
            print(f'Processed {count} {label_to_name[newLabel]} images')

    gc.collect()  # Call in the trashman to free some RAM
    # create labels for FK images, then extending that into labels
    labelsFK = [newLabel for i in range(count)]
    labels.extend(labelsFK)

    print(f'We have loaded {count} {label_to_name[newLabel]} images')
    print(f'rawDataset now has a total of {len(rawDataset)} images')

# takes in a tensor image of size (c, h, w) and show it
def showIm(image):
    img = image.transpose(0, 2) 
    img = img.transpose(0, 1) 
    imgplot = plt.imshow(torch.squeeze(img), cmap=plt.get_cmap('gray'))
    plt.show()

# Test function to print an image and its label from rawDataset and labels and label_to_name
def showImgFromRawDataSet(dataset, labels, index, label_to_name):
    #print(f'index is {index}')
    img = dataset[index]
    showIm(img)

    print(f'Label: {label_to_name[labels[index]]}')
    print('')

# Test to show random samples from a dataset and print it's label
def showRandomImgs(dataset, labels, num_samples,label_to_name):
    for i in range(num_samples):
        ind = random.randint(0,len(dataset)-1)
        showImgFromRawDataSet(dataset, labels, ind, label_to_name)
        
# Show a sample from a Pytorch Dataset class object
# Draws the first image of the sample then print its label
def showSample(data, index, label_to_name):
    sample = data[index]
    image = sample['image']

    showIm(image)

    label = sample['label']
    print(f'Label: {label_to_name[label]}')

def showNSKsamples(data, num_to_show):
    count = 0
    for i in range(len(data)):
        sample = data[i]
        label = sample['label']
        if label == 2:
            image = sample['image']
            showIm(image)
            count += 1
            print(f'Label: {label_to_name[label]}')
        if count >= num_to_show:
            break

# Get a free model from Pytorch
#   PT pre-made models:   https://pytorch.org/vision/0.8/models.html 
#   All pre-trained or vanilla models expect input images normalized in the same way, 
#   i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
#   where H and W are expected to be at least 224. 
#   The images have to be loaded in to a range of [0, 1] 
#       and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
def get_model(model_name, input_channels = 1, output_channels = 3, load_file_path = None):

    model = None

    if model_name.lower() == 'densenet121':
        model = models.densenet121()
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(1000, output_channels) # add another linear layer to output 3 classes 
        #print(denseNet121)
    elif model_name.lower() == 'densenet161':
        model = models.densenet161()
        model.features[0] = nn.Conv2d(input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(1000, output_channels) # add another linear layer to output 3 classes 
        #print(denseNet161)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18()
        model.conv1 = nn.Conv2d(input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, output_channels)
        #print(resnet18)
    else: 
        raise Exception(f'Error! Model not found')
        return None

    # If a pre_trained state dict is available, load it into model
    # Caution: State dict must be for the same model with the same input and output channels
    if load_file_path:
        state_dict = torch.load(model_save_path)
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return model
   
   
 
# Now for cross validation and training

# Workflow of training:
#   We start with trainDataPT and testDataPT, both Pytorch Dataset Objects
#   Create set of hyperparameters combinations C, with each C[i] = [batch_size, learning_rate] # more hyperpparamentes to be added
#   For each C[i] in C:
#       Initialize models with hyperparameters from C[i]
#       For each fold in k_fold:
#           Split trainDataPT into valset(1 fold) and trainset (k-1 folds)
#           Train trainset, save validation accuracy
#           Reset model weights
#       Calculate average accuracy for this combination C[i]
#   Using C[i] with highest accuracy, train all of trainDataPT, evaluate accuracy with testDataPT


# From that workflow, we have implementation notes:
# Implementation (high level): 
#     Manually create each combinations C we want to test (likely will automate this later)
#     c_accuracies = []
#     For each c in C:
#         criterion, optimizer = trainSetup(model, c)
#         c_accuracy = kfoldCV(model, criterion, optimizer, trainDataPT) # apply kfold cross validation, returns average accuracy over all folds
#         c_accuracies.append(c_accuracy)
#     c_best = C.index(max(c_accuracies))
#     criterion, optimizer = TrainSetup(model, c_best)     
#     loss, train_acc, test_acc = train(model, criterion, optimizer, trainDataPT, testDataPT, epochs) # train the model, returns lists for loss, train_acc, test_acc
#     plot loss, train_acc, test_acc vs epochs


# Fuctions that will be defined below and used, more details in each function's definitions below 
#
#     kfoldCV(model,criterion,optimizer,dataset, k_fold ,batch_size, numWorkers, epochs_each):
#         Used to apply kfoldCV, returns average accuracy over all folds
#         Uses train() to train each fold
#         Uses eval() to get accuracy of each fold
#         Uses weight_reset to reset weights of model after each fold
#         
#     train(model, criterion, optimizer, train_set, train_loader, test_loader, epochs, Print=False, Plot=False):
#         Trains the model
#         Used in kfoldCV to train a fold, later used to train the whole trainDataPT
#         Uses eval(model, train_loader) to get train accuracy each epoch
#         Uses eval(model, test_loader) to get test/validation accuracy each epoch
#         Returns lists: train_loss, train_acc, test_acc (loss and accuracy of each epoch)
#
#     eval(model, test_loader):
#         Used every time we want to get accuracy of model vs data
#         Runs the test_loader through the model
#         Calculates and returns accuracy of the model in that loader
#         
#     trainSetup(model, combination c):
#         Takes in a combination c = [batch_size, learning_rate, epochs_each_fold]
#         Returns criterion, optimizer
#         To be updated to set up more things using more hyperparameters
#
#     plotLossAcc(epochs, train_loss, train_score, val_score):
#         plots train_loss, train_score, val_score for each epoch
#
#     weight_reset(m)
#         resets weights in the model
#
#     
#         

# Several code cells below are these functions' definitions and details on their workflow    

# K-fold cross validation training function
# Code from https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
# Modified to fit this project

# Input:
# The function takes in all required objects for training: model, criterion, optimizer
#   As well as:
#     batch_size    batch size for PT loaders
#     dataset:      the training Pytorch dataset to be split
#     k_fold:       number of folds
#     numWorkers:   number of workers for the loaders
#     epoch:        number of epochs for each fold

# Output: 
#     Overall_acc:  the average accuracy of all the folds

# Workflow:
#   Calculate the segment size (size for each fold)
#   fold_acc = [] # accuracy of each fold
#   For i in k_fold:
#       Use fold size to determine indices to split dataset into train_set and val_set
#       Makes train_loader and val_loader
#       Calls train() to train each fold for epochs_each_fold times
#       fold_acc.append( eval(model, test_loader) ) to save the validation accuracy of each fold
#       Reset model weigths
#   return mean(fold_acc)

def kfoldCV(model, criterion, optimizer, dataset, k_fold ,batch_size, numWorkers, epochs_each, device):
    fold_acc = []

    # Calculate size of each segment/fold
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    
    for i in range(k_fold):
        #Calculate indices to split dataset
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        #msg
        #print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (trll,trlr,trrl,trrr,vall,valr))
        print(f'Training fold {i+1}/{k_fold}')
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        
        #print(len(train_set),len(val_set))
        #print()
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size,
                                          shuffle=True, num_workers = numWorkers)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size,
                                          shuffle=True, num_workers = numWorkers)
        
        # Train the fold
        train(model,criterion,optimizer, train_set, train_loader, val_loader, device, epochs = epochs_each, Print=True, Plot = True)

        # Evaluate the fold accuracy
        fold_acc.append(eval(model, val_loader, device))

        # Reset the model after the fold is done
        #print('Reseting weights')
        model.apply(weight_reset)

        # Clear GPU memory
        train_set = None
        val_set = None
        train_loader = None
        val_loader = None
        gc.collect()
    #print('Finished kfold cross validation (finally)!')

    # Clear cuda cache to free up some GPU RAM
    torch.cuda.empty_cache()
    # return average validation acc
    return sum(fold_acc) / len(fold_acc) 

# Training function
# Used to train 1 fold of the k-fold validation
# Also used for final training

# Input:
#   Variables for training: model, criterion, optimizer
#   train_set:      The training set, used to calculate accuracy
#   train_loader:   The Pytorch dataLoader object for the train set
#   val_loader:     The Pytorch dataLoader object for the validation set
#   test_loader:    The Pytorch dataLoader object for the test set
#   epochs:         Number of epochs to train
#   Print:          Whether we print loss, train acc and test acc each epoch 
#   Plot:           Whether we plot out loss, train acc and test acc vs epochs

# Output:
#   train_loss:     The training loss each epoch  
#   train_acc:      The training accuracy each epoch
#   test_acc:       The test accuracy each epoch
def train(model, criterion, optimizer, train_set, train_loader, val_loader, device, epochs=1, Print=False, Plot=False):
    train_loss =[]
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        losses = []
        correct = 0 # for training accuracy
        start = time.time()
        for i, sample in enumerate(train_loader):
            image = sample['image']
            label = sample['label']

            image = image.to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append loss 
            losses.append(loss.item())

            # Caculate train accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).float().sum()
        
        loss_epoch = sum(losses)/len(losses)
        # Append loss and various accuracies
        train_loss.append(loss_epoch)

        train_acc_epoch = correct / len(train_set)
        train_acc.append(train_acc_epoch)
        
        val_acc_epoch = eval(model, val_loader, device)
        val_acc.append(val_acc_epoch)
        if Print:
            end = time.time()
            print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss_epoch:.7f}, Accuracies: train: {train_acc_epoch:.6f}, validation/test: {val_acc_epoch:.6f}, train time(sec): {(end-start):.5f}')

        # Clear some gpu RAM after each epoch 
        gc.collect()
        torch.cuda.empty_cache()
    if Plot:
        plotLossAcc(epochs+1, train_loss, train_acc, val_acc)
            
    
    return train_loss, train_acc, val_acc

# Evaluation/Testing function

# Takes in the model and the Pytorch Dataloader for any set
# Tests the model using that set
# Returns the test accuracy in the range [0,1]
def eval(model, loader, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, sample in enumerate(loader):
            image = sample['image']
            label = sample['label']
            
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            n_samples += label.size(0)
            #print(label.size(0))
            n_correct += (predicted == label).sum().item()
            
        test_acc = n_correct / n_samples

        return test_acc
        
def trainSetup(model, C):
    batch_size = C[0]
    learning_rate = C[1]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return criterion, optimizer, batch_size
    
# Function to plot training loss, training accuracy, validation accuracy and test accuracy
def plotLossAcc(epochs, train_loss, train_score, val_score):
    x = list(range(1, epochs))
    #print(len(x))

    plt.figure(figsize=(6, 6))
    plt.plot(x, train_loss, label = "loss")

    plt.plot(x, train_score, label = "train accuracy")

    plt.plot(x, val_score, label = "validation/test accuracy")

    plt.xlabel('epochs')
    # Set the y axis label of the current axis.
    plt.ylabel('loss/accuracy')
    # Set a title of the current axes.
    plt.title('Loss/Accuracies each epoch')
    # show a legend on the plot
    plt.legend()

    # Display a figure.
    plt.show()


# Function to reinitialize values of the model
# usage: model.apply(weight_reset)
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        
