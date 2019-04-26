import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils import model_zoo
from torch import nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from PIL import Image
import os, sys, collections, random
from sklearn.metrics import precision_score
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import json, operator
import pdb
import matplotlib.pyplot as plt

device = torch.device('cuda')
batch_size = 64
num_epochs = 50
learning_rate = 0.02
label = [0]*20
dic = {}
for i in range(20):
    dic[i] = dict()
rank_dic = dic
t_lst, tailacc_lst = [], []

label_map = {'aeroplane':0 , 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
              'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
              'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
              'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

labels_map2 = {0: 'aeroplane', 1: 'bicycle',
              2: 'bird', 3: 'boat', 4: 'bottle',
              5: 'bus', 6: 'car', 7: 'cat',
              8: 'chair', 9: 'cow', 10: 'diningtable',
              11: 'dog', 12: 'horse', 13: 'motorbike',
              14: 'person', 15: 'pottedplant', 16: 'sheep',
              17: 'sofa', 18: 'train', 19: 'tvmonitor'}

test_trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])

trans_img = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])
    
# load data
class VOCDetection(Dataset):
    def __init__(self, root, image_set='train', transform):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        base_dir = 'VOCdevkit/VOC2012'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        ind_list = target['annotation']['object']
        ind = set()
        if (type(ind_list) is dict):
            ind.add(label_map[ind_list['name']])
        else:
            for e in ind_list:
                ind.add(label_map[e['name']])
            
        img = self.transform(img)
            
        return img, one_hot(list(ind))

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
# convert label to binary form    
def one_hot(label_list):
    opt = torch.zeros(20)
    for key in label_list:
        opt[key] = 1
    return opt


def train_model(model, optimizer, model_name, trainloader, valloader):
    best_loss = -1
    train_losses,val_losses,val_acc = [],[],[]
    criterion = nn.BCEWithLogitsLoss()

    for i in range(num_epochs):
        train_loss, val_loss, ap = 0.0, 0.0, 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad(): 
                model.eval()
                for images,labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    predictions = model(images)
                    val_loss += criterion(predictions, labels)
                    
                    predictions = predictions.cpu().detach().numpy().reshape(-1)
                    labels = labels.cpu().detach().numpy().reshape(-1)
                    ap += average_precision_score(labels, predictions)
                    
                    
            model.train()

        train_loss = train_loss/len(trainloader)
        val_loss = val_loss/len(valloader)
        ap = ap/len(valloader)

        if (best_loss<0) or (val_loss<best_loss):
            best_weights = model.state_dict()
            torch.save(best_weights, model_name+'.pkl')
            print('Better model found, saved as '+model_name+'.pkl')
            best_loss = val_loss
            

        print("Epoch: {}/{}.. ".format(i+1, num_epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Validation Loss: {:.3f}.. ".format(val_loss),
                  "Validation ap: {:.3f}..".format(ap))
                  
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acc.append(ap)



def training(trainloader, valloader):
    model = models.resnet18(pretrained = True)
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))

    model.fc = nn.Linear(512,20)

    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    
    model.to(device)
    train_model(model, optimizer, 'model', trainloader, valloader)

# loop through different t to find tailacc
def validating(model, t):
    criterion = nn.BCEWithLogitsLoss()
    ap, score, val_loss = 0.0, 0.0, 0.0
    output_list = []
    target_list = []
    with torch.no_grad(): 
        model = models.resnet18(pretrained = True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(model, map_location='cpu'))
        model.to(device)
        model.eval()
        for images,labels in valloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            val_loss += criterion(predictions, labels)
            
            output = torch.sigmoid(predictions)
            opt = (output > t)
            output_list.append(opt)
            target_list.append(labels)
                    
            predictions = predictions.cpu().detach().numpy().reshape(-1)
            labels = labels.cpu().detach().numpy().reshape(-1)
            ap += average_precision_score(labels, predictions)
                    
        opt = torch.cat(output_list,0)
        opt = opt.cpu().detach().numpy()
        tgt = torch.cat(target_list,0)
        tgt = tgt.cpu().detach().numpy()
        score = precision_score(tgt, opt, average='samples')
    print('score:' + str(score))
    print('t:' + str(t))
    t_lst.append(t)
    tailacc_lst.append(score)

# plot change of tailacc w.r.t t
def plot_t():
    plt.plot(t_lst, tailacc_lst)
    plt.xlabel('t')
    plt.ylabel('Tailacc')
    plt.savefig('tail_acc')
    plt.show()

# predic on a single image
def testing(model, path):
    result = []
    img = Image.open(path).convert('RGB')
    img = test_trans(img)
    img = img.unsqueeze(0)
    with torch.no_grad(): 
        model = models.resnet18(pretrained = True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(model , map_location='cpu'))
        model.eval()
        oup = model(img)
        output = torch.sigmoid(oup)
        opt = (output > 0.5)
        
        idx_lst = opt.tolist()[0]
        for i in range(20):
            if idx_lst[i]==1:
                result.append(labels_map2[i])
        return ' '.join(result)

# compute and rank by the prob of each image predicted
# and store the path to text file
def ranking(model):
    splits_dir = os.path.join('data/VOCdevkit/VOC2012', 'ImageSets/Main')
    split_f = os.path.join(splits_dir, 'val.txt')
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    images = [os.path.join('data/VOCdevkit/VOC2012/JPEGImages', x + ".jpg") for x in file_names]               
    
    with torch.no_grad(): 
        model = models.resnet18(pretrained = True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(model, map_location='cpu'))
        model.eval()
        model.to(device)
        for path in images:
            img = Image.open(path).convert('RGB')
            img = test_trans(img)
            img = img.unsqueeze(0)
            img.to(device)
            
            oup = model(img)
            output = torch.sigmoid(oup)
            opt = (output > 0.5)
            pred_classes = [i for i, x in enumerate(opt.tolist()[0]) if x==1]
            for pred_class in pred_classes:
                pred_prob = output[0].tolist()[pred_class]
                dic[pred_class][path] = pred_prob
    print('pred finish')
    for i in range(20):
        if dic[i]:
            sorted_x = sorted(dic[i].items(), key=operator.itemgetter(1), reverse = True)
            rank_dic[i] = sorted_x
    print('saving...')
    with open('rank.txt', 'w') as f:
        f.write(json.dumps(rank_dic))
    return rank_dic


if __name__ == '__main__':
    print('Start loading dataset...')
    train = VOCDetection(os.curdir, image_set='train', transform=trans_img)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True) 
    
    val = VOCDetection(os.curdir, image_set='val', transform=trans_img)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=True) 
    
    trainval = VOCDetection(os.curdir, image_set='trainval', transform=trans_img)
    trainvalloader = DataLoader(trainval, batch_size=batch_size, shuffle=True)


    print('Finish loading')
    print('==================================================')
    print('Start training model...')
    #testing(model)
    #training(trainloader, valloader)
    
    #tail_acc with different t:
    #t = 0.0
    #while t<= 1.0:
    #    validating('model_2116.pkl', t)
    #    t+= 0.05
    
    #plot_t()
    
        
        
        

        