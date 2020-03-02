#model.py
#Modified by ImKe on 2019/9/4.
#Copyright © 2019 ImKe. All rights reserved.

import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim, nn
import torch.nn.functional as F

import networks as nets

class Model:
    def __init__(self, hidden, learning_rate, batch_size):
        self.batch_size = batch_size
        self.net = nets.AutoEncoder(hidden)
        #self.opt = optim.Adam(self.net.parameters(), learning_rate)
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
        self.feature_size = hidden[0] # n_user/n_item

    def run(self, trainset, testlist, num_epoch, plot = True):
        RMSE = []
        for epoch in range(1, num_epoch + 1):
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True, pin_memory=True)
            self.train(train_loader, epoch)
            rmse = self.test(trainset, testlist,epoch)
            RMSE.append(rmse)
        if plot:
            x_label = np.arange(0,num_epoch,1)
            plt.plot(x_label, RMSE, 'b-.')
            my_x_ticks = np.arange(0, num_epoch, 10)
            plt.xticks(my_x_ticks)
            plt.title("RMSE of testing data")
            plt.xlabel("Number of epoch")
            plt.ylabel("RMSE")
            plt.grid()
            plt.show()


    #批训练
    def train(self, train_loader, epoch):
        self.net.train()
        features = Variable(torch.FloatTensor(self.batch_size, self.feature_size))
        masks = Variable(torch.FloatTensor(self.batch_size, self.feature_size))

        for bid, (feature, mask) in enumerate(train_loader):
            if mask.shape[0] == self.batch_size:
                features.data.copy_(feature)
                masks.data.copy_(mask)
            else:
                features = Variable(feature)
                masks = Variable(mask)
            self.opt.zero_grad()
            output = self.net(features)
            loss = F.mse_loss(output* masks, features* masks)
            loss.backward()
            self.opt.step()

        if (epoch%10==0):
            print ("Epoch %d, train end." % epoch)

    def test(self, trainset, testlist,epoch):
        self.net.eval()
        x_mat, mask, user_based = trainset.get_mat()
        features = Variable(x_mat)
        xc = self.net(features)
        if not user_based:
            xc = xc.t()
        xc = xc.cpu().data.numpy()

        rmse = 0.0
        for (i, j, r) in testlist:
            rmse += (xc[i][j]-r)*(xc[i][j]-r)
        rmse = math.sqrt(rmse / len(testlist))

        if (epoch%10==0):
            print (" Test RMSE = %f" % rmse)
        return rmse
