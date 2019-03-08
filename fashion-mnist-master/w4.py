# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:44:47 2019

@author: trying
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100,10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
def train_model(model,device,train_loader,optimizer,test_loader):
    best_loss=-1
    trainloss=list()
    testloss=list()
    for epoch in range(1,51):
        model.train()
        tl=0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data.resize_((data.shape[0],data.shape[2]**2))
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            tl=tl+float(loss.item())
        trainloss.append(tl/len(train_loader))
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data.resize_((data.shape[0],data.shape[2]**2))
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        testloss.append(test_loss)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if (best_loss<0) or test_loss<best_loss:
            bestweights=model.state_dict()
            torch.save(bestweights, 'fashionmnist_model.sd')
            print('saving model...')
            best_loss = test_loss
        print(trainloss,test_loss)
          
    
    plot(trainloss,testloss)

        #test the final model
    acc = 0
    dic = {}
    for i in range(10):
        dic[i] = [0, 0]
    model.load_state_dict(torch.load('fashionmnist_model.sd'))

    for data,labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        data.resize_((data.shape[0],data.shape[2]**2))
                
        log_ps = model(data)
        prob = torch.exp(log_ps)
        top_probs, top_classes = prob.topk(1, dim=1)
        equals = labels == top_classes.view(labels.shape)
        dic = compareTensor(top_classes.view(labels.shape), labels, dic)
        acc += equals.type(torch.FloatTensor).mean()
    print(acc/len(test_loader.dataset))
    displayClassRank(dic)

def plot(trainloss,  testloss):
    x=np.arange(1,len(trainloss)+1)
    plt.plot(x,trainloss,label="train loss")
    plt.plot(x,testloss,label="test loss")
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.show()
    
def compareTensor(t1, t2, dic):
    l1 = t1.tolist()
    l2 = t2.tolist()
    for i in range(len(l1)):
        dic[l2[i]][1] += 1
        if l1[i] == l2[i]:
            dic[l2[i]][0] += 1
    return dic

def displayClassRank(dic):
    for i in range(10):
        dic[i] = dic[i][0]*1.0/dic[i][1]
    sort = sorted(dic.items(), key=lambda kv: (-1)*kv[1])
    labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
    for item in sort:
        print('{} {:12}{:8}'.format(item[0], labels_map[item[0]], item[1]))
        
def main():
    
    # Training settings
    use_cuda = False

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")
    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=True, download=True,
                       transform=trans),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=False, transform=trans),
        batch_size=1000, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_model(model,device,train_loader,optimizer,test_loader)
    
        
if __name__ == '__main__':

    main()