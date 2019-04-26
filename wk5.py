import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import pdb

root_dir = 'flowersstuff/flowers_data/jpg'
train_name_file = 'flowersstuff/trainfile.txt'
val_name_file = 'flowersstuff/valfile.txt'
test_name_file = 'flowersstuff/testfile.txt'
valtest_name_file = 'flowersstuff/valtestfile.txt'
train_trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])
device = torch.device("cuda" )
print(device)
epoch = 20
lr = 0.03
save_model = False
criterion = nn.CrossEntropyLoss()
train_losses, val_losses, val_acc = [], [], []

def plot(model_name):
    plt.plot(train_losses, label = "Training losses")
    plt.plot(val_losses, label = "Validation losses")
    plt.legend()
    plt.xlabel('iteration')
    plt.savefig(model_name+ '_loss')
    plt.show() 
    
    plt.plot(val_acc, label = "Validation accuracy")
    plt.xlabel('iteration')
    plt.savefig(model_name+ '_valacc')
    plt.show()

class FlowerDataset(Dataset):
    def __init__(self, root_dir, filenames, transform):
        self.root_dir = root_dir
        self.filenames = filenames
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_name = self.filenames[idx].split()[0]
        labels = int(self.filenames[idx].split()[1])
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, labels
with open(train_name_file) as train_f:
    train_dataset = FlowerDataset(root_dir, train_f.readlines(), train_trans)
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
with open(val_name_file) as val_f:
    val_dataset = FlowerDataset(root_dir, val_f.readlines(), train_trans)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = True)
with open(test_name_file) as test_f:
    test_dataset = FlowerDataset(root_dir, test_f.readlines(), train_trans)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True)
print('finish loading dataset')

def train_model(model, optimizer, model_name):
    best_loss = -1
    for ep in range(epoch):
        train_loss, val_loss, acc = 0, 0, 0
        for images, labels in train_loader:
            pdb.set_trace()
            #images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # set gradient to 0
            op = model(images) # compute model prediction
            pdb.set_trace()
            loss = criterion(op, labels) # compute loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad(): 
                model.eval() # set to eval mode
                for images,labels in val_loader:
                    pdb.set_trace()
                    #images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)# compute loss in minibatch
                    
                    prob = torch.exp(log_ps)
                    top_probs, top_classes = prob.topk(1, dim=1)
                    equals = labels == top_classes.view(labels.shape)
                    acc += equals.type(torch.FloatTensor).mean()
            model.train() # set to train mode
        val_loss = val_loss/len(val_loader)
        if (best_loss<0) or (val_loss< best_loss):
            bestweights=model.state_dict()
            torch.save(bestweights, model_name+'.pkl')
            print('found better model, saving model...')
            best_loss = val_loss
        print("Epoch: {}/{}.. ".format(ep+1, epoch),
                  "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)), # loss avg over all minibatches
                  "Validation Loss: {:.3f}.. ".format(val_loss),
                  "Validation accuracy: {:.3f}..".format(acc/len(val_loader)))
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss)
        val_acc.append(acc/len(val_loader))
    print(train_losses)
    print(val_losses)
    print(val_acc)
    plot(model_name)

def train_model_1():
    model = models.resnet18(pretrained = False)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr)
    train_model(model, optimizer, 'model_1') 
#train_model_1()
    
def train_model_2():
    model = models.resnet18(pretrained = True)
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    #model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr)
    train_model(model, optimizer, 'model_2')  
train_model_2()

def train_model_3():
    model = models.resnet18(pretrained = True)
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    optimizer = optim.SGD([
            {'params': model.layer4.parameters()},
            {'params': model.fc.parameters()}
            ],lr=lr)
    model.to(device)
    train_model(model, optimizer, 'model_3')    
#train_model_3()        
    
def test(model_path):
    acc = 0
    dic = {}
    for i in range(10):
        dic[i] = [0, 0]
    model = models.resnet18(pretrained = False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    for images,labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): 
            log_ps = model(images)
            prob = torch.exp(log_ps)
            top_probs, top_classes = prob.topk(1, dim=1)
            equals = labels == top_classes.view(labels.shape)
            acc += equals.type(torch.FloatTensor).mean()
    print(acc/len(test_loader))
#test('model_1.pkl')  
    
 
    
    
    
    
    
    
    
    
    