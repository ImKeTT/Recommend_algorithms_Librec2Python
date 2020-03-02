#auther: unkonwn
#date:2019/11
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import *


class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=200, num_batches=10,
                 batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_C = None  # Movie feature vectors
        self.w_I = None  # User feature vectors

        self.err_train = []
        self.err_val = []
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_rmse = []
        self.test_rmse = []

    def fit(self, train_vec, val_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  

        pairs_tr = train_vec.shape[0] 
        pairs_va = val_vec.shape[0] 

        # 1-p-i, 2-m-c

        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(val_vec[:, 0]))) + 1  
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(val_vec[:, 1]))) + 1  

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_item, self.num_feat)  
            self.w_I = 0.1 * np.random.randn(num_user, self.num_feat)  

            self.w_C_inc = np.zeros((num_item, self.num_feat))  
            self.w_I_inc = np.zeros((num_user, self.num_feat))  

        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  
            np.random.shuffle(shuffled_order)  

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)
                batch_idx = np.mod(np.arange(self.batch_size * batch, self.batch_size * (batch + 1)),
                                   shuffled_order.shape[0])  
                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID, :], self.w_C[batch_comID, :]),
                                  axis=1)  # mean_inv subtracted
                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID, :]) + self._lambda * self.w_C[
                                                                                                         batch_comID, :]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID, :]) + self._lambda * self.w_I[
                                                                                                         batch_invID, :]

                dw_C = np.zeros((num_item, self.num_feat))
                dw_I = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i], :] += Ix_C[i, :]
                    dw_I[batch_invID[i], :] += Ix_I[i, :]

                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size
                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 + 0.5 * self._lambda * (LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)
                    self.err_train.append(np.sqrt(obj / pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(val_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr) / np.sqrt(pairs_va))

                    # Print info
                if batch == self.num_batches - 1:
                    print('Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1]))
                    self.train_rmse.append(self.err_train[-1])
                    self.test_rmse.append(self.err_val[-1])

    def predict(self, invID):
        return np.dot(self.w_C, self.w_I[int(invID), :]) + self.mean_inv 

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, model, test_vec, k=10):  
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)

