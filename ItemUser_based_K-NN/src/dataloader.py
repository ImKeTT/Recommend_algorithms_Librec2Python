#dataloader.py
#Created by ImKe on 2019/12.
#Copyright © 2019 ImKe. All rights reserved.

import os
import sys

path_prefix = '../ml-100k/'
def LoadMoviLensFile(dataset, Userbased = True):
    filename = path_prefix + dataset 
    prefer = {}
    for line in open(filename, 'r'):
        (UserID, MovieID, rating, timestamp) = line.split('\t')
        if (Userbased == True):
            prefer.setdefault(UserID, {})
            prefer[UserID][MovieID] = float(rating)

        else:
            prefer.setdefault(MovieID, {})
            prefer[MovieID][UserID] = float(rating)

    return prefer
    