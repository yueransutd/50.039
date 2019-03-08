# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pdb

batch_size = 64
epoch = 50
lr = 0.01
momentum = 0.5
save_model = False
train_losses, test_losses = [], []
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

def train_model():
    best_loss = -1
    for ep in range(epoch):
        train_loss, val_loss = 0,0
        for images, labels in trainloader:
            optimizer.zero_grad() # set gradient to 0
            op = model(images) # compute model prediction
            loss = criterion(op, labels) # compute loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        # validation
        else:
            with torch.no_grad(): 
                model.eval() # set to eval mode
                for images,labels in testloader:
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)# compute loss in minibatch
                    
            model.train() # set to train mode
        val_loss = val_loss/len(testloader)
        if (best_loss<0) or (val_loss< best_loss):
            bestweights=model.state_dict()
            torch.save(bestweights, 'fashionmnist_model.sd')
            print('found better model, saving model...')
            best_loss = val_loss
        print("Epoch: {}/{}.. ".format(ep+1, epoch),
                  "Training Loss: {:.3f}.. ".format(train_loss/len(trainloader)), # loss avg over all minibatches
                  "Validation Loss: {:.3f}.. ".format(val_loss))
        train_losses.append(train_loss/len(trainloader))
        test_losses.append(val_loss)
    plot()
    return bestweights
        

def test():
    acc = 0
    dic = {}
    for i in range(10):
        dic[i] = [0, 0]
    model.load_state_dict(torch.load('fashionmnist_model.sd'))
    for images,labels in testloader:
        log_ps = model(images)
        prob = torch.exp(log_ps)
        top_probs, top_classes = prob.topk(1, dim=1)
        equals = labels == top_classes.view(labels.shape)
        dic = compareTensor(top_classes.view(labels.shape), labels, dic)
        acc += equals.type(torch.FloatTensor).mean()
    print(acc/len(testloader))
    displayClassRank(dic)
    

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
    for item in sort:
        print('{} {:12}{:8}'.format(item[0], labels_map[item[0]], item[1]))


def plot():
    plt.plot(train_losses, label = "Train losses")
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.savefig('train_loss')
    plt.show()
    plt.plot(test_losses, label = "Test losses")
    plt.xlabel('iteration')
    plt.ylabel('val loss')
    plt.savefig('val_loss')
    plt.show()

class FashionData():
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=False, train=True, transform = self.transform)
        self.testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train = False, transform=self.transform, download = False)
    def __len__():
        return len(self.trainset)
    def __getitem__(self, idx):
        return self.trainset[idx]
        
        
fashionData = FashionData() 
trainloader = torch.utils.data.DataLoader(fashionData.trainset, batch_size=64, shuffle=True) 
testloader = torch.utils.data.DataLoader(fashionData.testset, shuffle=False, batch_size=64)
model = Net()
optimizer = optim.SGD(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

train_model()    
test()













