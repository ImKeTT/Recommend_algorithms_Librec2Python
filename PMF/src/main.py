#main.py
#Created by ImKe on 2019/11.
#Copyright © 2019 ImKe. All rights reserved.

from daraloader import load_rating_data, spilt_rating_dat
from PMF import *

if __name__ == "__main__":
    pmf = PMF()
    ratings = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = spilt_rating_dat(ratings)
    pmf.fit(train, test)

    plt.plot(range(pmf.maxepoch), pmf.train_rmse, 'b-.', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.test_rmse, 'go-', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()