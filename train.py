
import time
import copy
import os
import pandas as pd

from pathlib import Path
from PIL import Image
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
plt.style.use("ggplot")


########################################################################
###################construct the argument parser########################

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs to train our network for')
parser.add_argument('-b', '--batch', type=int, default=4,
    help='number of training examples to be considered as a  batch')
args = vars(parser.parse_args())

epochs = args['epochs']
batch_size= args['batch']

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'data/Cars'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####fine tuning Swin Transformer#################

resnet =  models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


num_classes = 2
for param in resnet.parameters():
        param.requires_grad=False

    #Training the last fc layer
in_features =  resnet.fc.in_features
resnet.fc = nn.Linear(in_features,num_classes)
            
'''
for param in resnet.parameters():
    if param.requires_grad == True :
        print(param.shape)
'''    


model = resnet.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)


####### Training Script#########
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    train_loss_array=[]
    train_accuracy_array=[]
    test_loss_array=[]
    test_accuracy_array=[]

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_loss_array.append(round(epoch_loss,3))
                train_accuracy_array.append(round(epoch_acc.item(),3))
            if phase == 'test':
                test_loss_array.append(round(epoch_loss,3))
                test_accuracy_array.append(round(epoch_acc.item(),3))

            print(f'{phase} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.3f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print("Saving model ...")
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    with open("metrics.txt", "w") as outfile:
        outfile.write("Accuracy: " + str(round(best_acc.item(),3)) + "\n")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_loss_array,train_accuracy_array,test_loss_array,test_accuracy_array

model_ft,train_loss_array,train_accuracy_array,test_loss_array,test_accuracy_array = train_model(model, loss_fn , opt, exp_lr_scheduler,
                       epochs)


#######log test loss to loss.csv
logs = pd.DataFrame({'train_loss': train_loss_array ,'test_loss':test_loss_array,'train_accuracy':train_accuracy_array,'test_accuracy':test_accuracy_array})
logs['test_accuracy'].to_csv("logs.csv",index=False)

######## save train loss and test loss plot 
fig = px.line(logs, y=logs.columns[:2], markers=True)
fig.update_layout(yaxis_range=[0,1])
fig.write_image("loss.png")



#######generate classes.csv#########################
def evaluate_model(model):
    predicted = []
    actual = []
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions= preds.squeeze().detach().cpu().numpy()
            target = labels.squeeze().detach().cpu().numpy()
            [predicted.append("audi") if i == 0 else predicted.append("toyota")  for i in predictions]
            [actual.append("audi") if i == 0 else actual.append("toyota")  for i in target]
            
        return predicted , actual
predicted,actual  = evaluate_model(model_ft)


import pandas as pd
classes = pd.DataFrame({'predicted': predicted ,'actual':actual})
classes.to_csv("classes.csv",index=False)




############ plot confusion matrix################
plt.figure(figsize=(7,5))
data = confusion_matrix(classes['actual'], classes['predicted'])
df_cm = pd.DataFrame(data, columns=np.unique(actual), index = np.unique(actual))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True)
plt.savefig('confusion_matrix.png')

