from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import pdb
import torch
import unicodedata
import string
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim
from torch.autograd import Variable



def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
# all_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;''
n_letters = len(all_letters)
# 57

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines_train,  category_lines_test = {}, {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0] # Arabic, Chinese...
    all_categories.append(category) #list: ['Arabic', 'Chinese'...]
    lines = readLines(filename)
    length = len(lines)
    train_lines = lines[:int(length*0.7)]
    test_lines = lines[int(length*0.7):]
    category_lines_train[category] = train_lines #dict: {'Arabic':[...], 'Chinese:[...]'}
    category_lines_test[category] = test_lines
    
n_categories = len(all_categories) #18

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)#57)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, memory):
        #x = x.unsqueeze(0)
        #batch_in = torch.zeros((self.batch_size, self.num_layers, self.input_size))
        batch_in = Variable(x)
        pack = torch.nn.utils.rnn.pack_sequence(batch_in)
        
        output, (hn, cn)= self.lstm(pack, (hidden,memory))
        #unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

        batch_tensors = []
        for i in range(self.batch_size):
            op = self.fc(unpacked[0][i].unsqueeze(0).unsqueeze(0)[:, -1, :])
            op = self.softmax(op)
            batch_tensors.append(op)
        out = torch.cat((tuple(batch_tensors)), 0)
        return out, (hn, cn)
        #output: [batch, 18], hn : [1, batch, 128]
    #init memory cell and hidden state of the LSTM to zero
    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
    def initMemCell(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

n_hidden = 128
n_layers = 1
batch_size = 1
rnn = RNN(n_letters, n_hidden, n_layers, n_categories, batch_size)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

import random

def randomChoice(l): #l= language list
    return l[random.randint(0, len(l) - 1)] # language name

def randomTrainingExample(training):
    category = randomChoice(all_categories)
    if training:
        line = randomChoice(category_lines_train[category])
    else: 
        line = randomChoice(category_lines_test[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# training
lr = 0.02
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=lr)


n_iters = 100000
print_every = 5000
plot_every = 1000


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
start = time.time()

def training():
    all_losses, iter_lst_train, iter_lst_test, val_losses = [], [], [], []
    current_loss = 0
    cur_test_iter = 0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(True)
        hidden = rnn.initHidden()
        memory = rnn.initMemCell()
        
        rnn.zero_grad()
        for i in range(0, line_tensor.size()[0], batch_size):
            if len(line_tensor[i:i+batch_size])==batch_size:
                output, _ = rnn(line_tensor[i:i+batch_size], hidden, memory)
        #print(type(output), type(torch.FloatTensor(category_tensor).repeat(2).view(2, -1)))
        cat_t = torch.Tensor(category_tensor.type_as(torch.FloatTensor())).repeat(batch_size).view(batch_size, -1).squeeze(-1)
        loss = criterion(output, cat_t.long())
        loss.backward()
        optimizer.step()
        #output, loss = train(category_tensor, line_tensor)
        current_loss += loss.item()
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output) 
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
            iter_lst_train.append(iter)
            
            # validate
            val_loss= 0
            acc = 0
            with torch.no_grad():
                rnn.eval()
                
                for iterate in range(plot_every):
                    category, line, category_tensor, line_tensor = randomTrainingExample(False)
                    hidden = rnn.initHidden()
                    memory = rnn.initMemCell()
                    for i in range(line_tensor.size()[0]):
                        output, _ = rnn(line_tensor[i], hidden, memory)
                    loss = criterion(output, category_tensor)
                    val_loss += loss.item()
                    top_probs, top_classes = output.topk(1, dim=1)
                    equals = category_tensor == top_classes.view(category_tensor.shape)
                    acc += equals.type(torch.FloatTensor).mean()
                    if iterate % app_every == 0 and iterate!= 0 :
                        val_losses.append(val_loss / app_every)
                        val_loss = 0
                        iter_lst_test.append(iterate+ cur_test_iter)
                cur_test_iter+= plot_every
            
    print(acc/plot_every)
    plot_graph(iter_lst_test, val_losses, 'val_loss', 'val_loss')
    plot_graph(iter_lst_train, all_losses, 'train_loss', 'train_loss')
            
n_confusion = 10000
app_every = 200
def validate():
    acc = 0
    val_loss = 0.0
    val_losses, iter_lst = [], []
    with torch.no_grad():
        rnn.eval()
        
        for iterate in range(n_confusion):
            category, line, category_tensor, line_tensor = randomTrainingExample(False)
            hidden = rnn.initHidden()
            memory = rnn.initMemCell()
            for i in range(0, line_tensor.size()[0], batch_size):
                if len(line_tensor[i:i+batch_size])==batch_size:
                    output, _ = rnn(line_tensor[i:i+batch_size], hidden, memory)
            cat_t = torch.Tensor(category_tensor.type_as(torch.FloatTensor())).repeat(2).view(2, -1).squeeze(-1)
            loss = criterion(output, cat_t.long())
            val_loss += loss.item()
            top_probs, top_classes = output.topk(1, dim=1)
            equals = category_tensor == top_classes.view(category_tensor.shape)
            acc += equals.type(torch.FloatTensor).mean()
            if iterate % app_every == 0:
                val_losses.append(val_loss / app_every)
                val_loss = 0
                iter_lst.append(iterate)
    plot_graph(iter_lst, val_losses, 'val_loss', 'val_loss')
    print(acc/n_confusion)
    
def plot_graph(iter_lst, lst, file_name, y_name):
    plt.plot(iter_lst, lst)
    plt.xlabel('iteration')
    plt.ylabel(y_name)
    plt.savefig(file_name)
    plt.show()

training()
#validate()



