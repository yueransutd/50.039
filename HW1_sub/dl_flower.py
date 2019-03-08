import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from random import shuffle
import math

feats_path= 'flowers17/feats'
train_path = 'train.npy'
val_path = 'val.npy'
test_path = 'test.npy'
c_list = [0.01, 0.1, math.pow(0.1, 0.5), 1, math.pow(10, 0.5), 10, math.pow(100, 0.5)]
acc_list = [[] for i in range(7)]

def split_train_test():
    train, val, test = [], [], []
    all_files = os.listdir(os.path.abspath('flowers17/feats'))
    all_feats = [[] for i in range(17)]
    for name in all_files:
        class_idx = int((int(name[6:10])-1)/80)
        arr = np.load(feats_path + '/' + name)
        all_feats[class_idx].append(np.append(arr, class_idx).reshape((1,513)))
    for idx in range(17):
        shuffle(all_feats[idx])
        train += all_feats[idx][:40]
        val += all_feats[idx][40:60]
        test += all_feats[idx][60:]
    
    train_ = np.concatenate(train)
    np.save(train_path, train_)
    val_ = np.concatenate(val)
    np.save(val_path, val_)
    test_ = np.concatenate(test)
    np.save(test_path, test_)

#*** train on training set and test on validation***
def train(c):
    all_proba = []
    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)
    x_train, y_train = np.split(train, (512,), axis = 1)
    # x_train (680, 512), y_train (680, 1)
    x_val, y_val = np.split(val, (512,), axis = 1)
    x_test, y_test = np.split(test, (512,), axis = 1)
    
    for idx in range(17):
        y_train1 = (y_train==idx).astype(float)
              
        clf = SVC(kernel = 'linear', probability = True, C= c)
        clf.fit(x_train, y_train1)
        
        proba = clf.predict_proba(x_val)
        all_proba.append(proba)
    
    pred_class = []
    ll = []
    for i in range(340):
        l = []
        for j in range(17):
            l.append(all_proba[j][i][1])
        ll.append(l)
    for idex in range(340):
        pred_idx = ll[idex].index(max(ll[idex]))
        pred_class.append(pred_idx)
    
    acc = accuracy_score(y_val, pred_class)
    print(acc)
    return acc

def find_c():
    # get the avg over 5 iters
    for itr in range(5):   
        for ci in range(7):   
            acc = train(c_list[ci])
            acc_list[ci].append(acc)
    print([sum(l)/len(l) for l in acc_list])
    # [0.9294117647058823, 0.9254901960784313, 0.926470588235294, 0.9284313725490195, 0.9284313725490195, 0.9245098039215686, 0.9245098039215686]
    # the highest value correspond to C=0.01


#*** train the model again using training+ validation***
def train_whole():
    all_proba = []
    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)
    x_train, y_train = np.split(train, (512,), axis = 1)
    x_val, y_val = np.split(val, (512,), axis = 1)
    x_test, y_test = np.split(test, (512,), axis = 1)
    x_train_whole = np.concatenate([x_train, x_val], axis = 0)
    y_train_whole = np.concatenate([y_train, y_val], axis = 0)
    
    for idx in range(17):
        y_train1 = (y_train_whole==idx).astype(float)
              
        clf = SVC(kernel = 'linear', probability = True, C= 0.01)
        clf.fit(x_train_whole, y_train1)
        
        proba = clf.predict_proba(x_test)
        all_proba.append(proba)
        
    pred_class = []
    ll = []
    for i in range(340):
        l = []
        for j in range(17):
            l.append(all_proba[j][i][1])
        ll.append(l)
    for idex in range(340):
        pred_idx = ll[idex].index(max(ll[idex]))
        pred_class.append(pred_idx)
        
    acc = accuracy_score(y_test, pred_class)
    print(confusion_matrix(y_test, pred_class))
    print(acc)
    return acc
      
train_whole()
    
