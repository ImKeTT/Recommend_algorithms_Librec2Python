import numpy
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile
import os
import sys

def load_data():
    max_item_id  = -1
    train_history = {}
    with open('/Users/imke/Downloads/Auto_Rec_github/AutoRec-Pytorch/ml-100k/ua.base', 'r') as file:
        for line in file:
            user_id, item_id, rating, timestamp = line.encode('utf-8').decode('utf-8').rstrip().split('\t')
            if int(user_id) not in train_history:
                train_history[int(user_id)] = [int(item_id)]
            else:
                train_history[int(user_id)].append(int(item_id))

            if max_item_id < int(item_id):
                max_item_id = int(item_id)

    test_history = {}
    with open('/Users/imke/Downloads/Auto_Rec_github/AutoRec-Pytorch/ml-100k/ua.test', 'r') as file:
        for line in file:
            user_id, item_id, rating, timestamp = line.encode('utf-8').decode('utf-8').rstrip().split('\t')
            if int(user_id) not in test_history:
                test_history[int(user_id)] = [int(item_id)]
            else:
                test_history[int(user_id)].append(int(item_id))

    max_item_id += 1 # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)#one_hot pic
        train_x[i] = numpy.sum(mat, axis=0) #每一列的元素相加，压缩为一列

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
