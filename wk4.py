# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
import os
import getimagenetclasses
from PIL import Image
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch import nn
import pdb

text_dir = 'val'
root_dir = 'imagespart'
filenames = [os.path.splitext(filename)[0] for filename in os.listdir('imagespart')]

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, text_dir, root_dir, filenames, transform):
        self.root_dir = root_dir
        self.text_dir = text_dir
        self.filenames = filenames
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        img_name = os.path.join(self.root_dir, file_name+ '.jpeg')
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image)
        labels,_,__ = getimagenetclasses.test_parseclasslabel(os.path.join(self.text_dir, file_name+ '.xml'))
        return image, labels

def getTransform(normalize, multiCrop):
    if normalize== False: #task 1 not normalize
        train_trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])
    elif normalize: #task 1 with normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_trans = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    elif multiCrop ==5: #task 2 five crops
        train_trans = transforms.Compose([transforms.Resize(280),
                                    transforms.FiveCrop(224),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    elif multiCrop ==10: #task 2 ten crops
        train_trans = transforms.Compose([transforms.Resize(280),
                                    transforms.TenCrop(224),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    else: # task 3
        train_trans = transforms.Compose([transforms.Resize(400),
                                    transforms.FiveCrop(330),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

    return train_trans



def task1(train_trans, normalize):
    imageNetDataset = ImageNetDataset(text_dir, root_dir, filenames, train_trans)
    imageNetLoader = torch.utils.data.DataLoader(imageNetDataset, batch_size = 250, shuffle = True)
    model = models.resnet18(pretrained = True)
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    images, labels = next(iter(imageNetLoader))
    outputs = model(torch.autograd.Variable(images))
    value, index = torch.max(outputs, 1)
    score = torch.sum (torch.eq (labels,index) )
    acc = score.item()/len(labels)
    if not normalize:
        print('Task1 no normalization:', acc)
    else:
        print('Task1 with normalization:', acc)

#task1(getTransform(True, None), True)
#task1(getTransform(False, None), False)
        
def task2(train_trans, multiCrop):
    imageNetDataset = ImageNetDataset(text_dir, root_dir, filenames, train_trans)
    imageNetLoader = torch.utils.data.DataLoader(imageNetDataset, batch_size = 250, shuffle = True)
    model = models.resnet18(pretrained=True)
    images, labels = next(iter(imageNetLoader))
    
    batchsize, ncrops, c, h, w = images.size()
    _images = images.view(-1,c,h,w)
    outputs = model(torch.autograd.Variable(_images))
    
    outputs_avg = outputs.view(batchsize, ncrops, -1).mean(1)
    value, index = torch.max(outputs_avg,1)
    score = torch.sum(torch.eq(labels,index))
    acc = score.item() / len(labels)
    if multiCrop == 5:
        print("Task 2 FiveCrop: ", acc)
    elif multiCrop == 10:
        print("Task 2 TenCrop: ", acc)
        
#task2(getTransform(None, 5), 5)    
#task2(getTransform(None, 10), 10)   
        
def task3(train_trans):
    imageNetDataset = ImageNetDataset(text_dir, root_dir, filenames, train_trans)
    imageNetLoader = torch.utils.data.DataLoader(imageNetDataset, batch_size = 128, shuffle = True)
    # resnet modification
    model = models.resnet18(pretrained=True)
    model.avgpool=torch.nn.AdaptiveAvgPool2d(1)
    # squeezenet modification
    model_sn = models.squeezenet1_0(pretrained=True)
    model_sn.classifier[3]=torch.nn.AdaptiveAvgPool2d(1)
    
    images, labels = next(iter(imageNetLoader))
    batchsize, ncrops, c, h, w = images.size()
    _images = images.view(-1,c,h,w)
    outputs = model(torch.autograd.Variable(_images))
    outputs_avg = outputs.view(batchsize, ncrops, -1).mean(1)
    value, index = torch.max(outputs_avg,1)
    score = torch.sum(torch.eq(labels,index))
    acc = score.item() / len(labels)
    print('Task 3 resnet18:' , acc)
    
    outputs_sn = model_sn(torch.autograd.Variable(_images))
    outputs_avg_sn = outputs_sn.view(batchsize, ncrops, -1).mean(1)
    value, index = torch.max(outputs_avg_sn,1)
    score = torch.sum(torch.eq(labels,index))
    acc = score.item() / len(labels)
    print('Task 3 squeezenet:', acc)


#task3(getTransform(None, None))

    
    