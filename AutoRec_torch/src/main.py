#main.py
#Modified by ImKe on 2019/9/5.
#Copyright © 2019 ImKe. All rights reserved.

import setdata as sd
import model
from datetime import datetime
from data_loader import load_data

# parameters
rank = 100
batch_size = 128
epoch_num = 500
plotbool = True
user_based = False

if __name__ == '__main__':
    start = datetime.now()
    train_list, test_list, n_user, n_item = load_data('ratings', 0.9)
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user

    setmod = model.Model(hidden=[h, rank*3],
                      learning_rate=0.2,
                      batch_size=batch_size)

    RMSE = setmod.run(trainset, test_list, num_epoch = epoch_num, plot = plotbool)

    end = datetime.now()
    print ("Total time: %s" % str(end-start))


