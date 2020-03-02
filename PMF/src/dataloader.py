#dataloader.py
#Created by ImKe on 2019/10.
#Copyright © 2019 ImKe. All rights reserved.

from numpy import *
import random

path_prefix = '../ml-1M'
def load_rating_data(dataset = 'ratings'):
    prefer = []
    filename = path_prefix+dataset+'.dat'
    for line in open(filename, 'r'):  
        (userid, movieid, rating, ts) = line.split("::")  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    return data

#不同于auto_rec和K-NN对于数据的处理
#这里以1:5的比例将ratings分为测试集与训练集
def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = array(train_data)
    test_data = array(test_data)
    return train_data, test_data