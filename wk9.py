import torch
import torch.nn as nn
import string
import csv
import pdb
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import random

all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters) + 1 # Plus EOS marker
#78
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st']=[]
    filterwords=['NEXTEPISODE']
    with open('star_trek_transcripts_all_episodes_f.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
                    
                    v=el.strip().replace(';','').replace('\"','') #.replace('=','') #.replace('/',' ').replace('+',' 
                    category_lines['st'].append(v)
    return category_lines,all_categories
            #dict, key = 'st', value = []   ; list ['st'] 60084
            
# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor 
 
    # LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomExample(training):
    c_l, c = get_data()
    train_lines = c_l['st'][:50000]
    test_lines = c_l['st'][50000:]

    if training:
        input_line_tensor_lst = [inputTensor(train_line) for train_line in train_lines]
        target_line_tensor_lst = [targetTensor(train_line) for train_line in train_lines]    
    else:
        input_line_tensor_lst = [inputTensor(test_line) for test_line in test_lines]
        target_line_tensor_lst = [targetTensor(test_line) for test_line in test_lines]    
    return input_line_tensor_lst, target_line_tensor_lst
    # torch.Size([sentence_len, 1, 59]); len(target_line_tensor) = 83


n_categories = 1
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, memory):
        x = x.unsqueeze(0)
        output, (hn, cn)= self.lstm(x, (hidden, memory))
        output = self.fc(output[:, -1, :])
        output = self.dropout(output)
        output = self.softmax(output/ 0.5)
        return output, (hn, cn)

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
    def initMemCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


learning_rate = 0.005
n_hidden = 100
op_size = 78
n_layers = 2
rnn = RNN(n_letters, n_hidden, n_layers, op_size)
rnn.load_state_dict(torch.load('mymodel.pt'))
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    
    hidden = rnn.initHidden()
    memcell = rnn.initMemCell()
    rnn.to(device)
    input_line_tensor = input_line_tensor.to(device)
    target_line_tensor = target_line_tensor.to(device)

    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        hidden, memcell = hidden.to(device), memcell.to(device)
        output, (hidden, memcell) = rnn(input_line_tensor[i], hidden, memcell)
        l = criterion(output, target_line_tensor[i])
        loss += l
    loss.backward()
    optimizer.step()
    return output, loss.item() / input_line_tensor.size(0)

def evaluate(inp_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)

    hidden = rnn.initHidden()
    memcell = rnn.initMemCell()
    
    inp_line_tensor = inp_line_tensor.to(device)
    target_line_tensor = target_line_tensor.to(device)
    rnn.to(device)
    corr = 0
    loss = 0
    for i in range(inp_line_tensor.size()[0]):
        hidden, memcell = hidden.to(device), memcell.to(device)
        output, (hidden, memcell)  = rnn(inp_line_tensor[i], hidden, memcell)
        l = criterion(output, target_line_tensor[i])
        loss += l
        topv, topi = output.topk(1)
        topi = topi[0][0]
        if topi==target_line_tensor[i]:
            corr += 1
        
    return output, loss.item() / inp_line_tensor.size(0), 1.0*corr/ inp_line_tensor.size()[0]

max_length  = 180
def sample(start_letter):
    with torch.no_grad():
        inp = inputTensor(start_letter)
        hidden = rnn.initHidden()
        memcell = rnn.initMemCell()

        rnn.to('cpu')
        output_name = start_letter
        for i in range(max_length):
            output, (hidden, memcell) = rnn(inp[0], hidden, memcell) #input.shape([1,1,59])
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1: #reach EOS
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            inp = inputTensor(letter)
        return output_name
    
# Get multiple samples from one category and multiple starting letters
def samples():
    start_letters = ''.join(random.choice('WERTUIOPASDFGHJKLZCVBNM') for x in range(random.randint(10, 20)))
    for start_letter in start_letters:
        with open('out.txt', 'a') as f:
            f.write(sample(start_letter))    
        print(sample(start_letter))

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def plot_graph(lst, file_name, y_name):
    plt.plot(lst)
    plt.xlabel('epoch')
    plt.ylabel(y_name)
    plt.savefig(file_name)
    plt.show()

train_losses, test_losses, test_accs = [], [], []
train_loss, test_loss = 0, 0 # Reset every plot_every iters
best_loss = -1
start = time.time()
iter = 0
test_acc = 0
epoch = 0
n_epochs = 8
sample_iter = 0
# for every epoch:
for epoch in range(n_epochs):
    epoch += 1
    print('current epoch: %s' % (epoch))
    train_len = len(randomExample(training = True)[0])
    for line_tensor in zip(randomExample(training = True)[0], randomExample(training = True)[1]):
        tr_output, tr_loss = train(line_tensor[0], line_tensor[1])
        train_loss += tr_loss
        sample_iter += 1
        
        if sample_iter%5000 == 0:
            print(sample_iter, train_loss/ sample_iter)
            samples()   
    train_losses.append(train_loss / train_len)
    print('%s epoch = %s, training loss = %.4f' % (timeSince(start), epoch, train_loss/train_len))
    train_loss = 0
    sample_iter = 0
    samples()
    # test
    with torch.no_grad():
        rnn.eval()
        test_len = len(randomExample(training = False)[0])
        for line_tensor_ in zip(randomExample(training = False)[0], randomExample(training = False)[1]):
            tst_output, tst_loss, acc = evaluate(line_tensor_[0], line_tensor_[1])
            test_loss += tst_loss
            test_acc += acc
            # calculate acc
        # save model
        if best_loss== -1 or test_loss< best_loss:
            print('find better model, saving...')
            best_loss = test_loss
            best_model_wts = rnn.state_dict()
            torch.save(best_model_wts, 'mymodel.pt')
        test_losses.append(test_loss / test_len)
        test_accs.append(test_acc)
        print('epoch = %s, test loss = %.4f, accuracy = %.4f' % 
              (epoch, 
               test_loss / test_len,
               test_acc/ test_len))
        test_loss = 0
        test_acc = 0
    rnn.train()
    
print('train loss: '+ str(train_losses))   
print('test loss: '+ str(test_losses))               
print('test acc: '+ str(test_accs))               

plot_graph(train_losses, 'train_loss', 'train_loss')
plot_graph(test_losses, 'test_loss', 'test_loss')


        
        
        


