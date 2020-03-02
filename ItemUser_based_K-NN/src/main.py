#main.py
#Created by ImKe on 2019/12.
#Copyright © 2019 ImKe. All rights reserved.

if __name__ == "__main__":
    print("The data of the upcoming test is ml-100k")
    starttime = datetime.datetime.now()
    print("%3s%20s%20s%20s" % ('K', "RMSE","MAE","time"))
    ran = [10, 20, 30, 40, 50, 60, 70, 80]
    rmse0 = []
    mae0 = []

    for k in ran:
        r = getAllUserRating(k)
        rmse = getRMSE(r)
        rmse0.append(rmse)
        mae = getMAE(r)
        mae0.append(mae)
        print("%3d%19.3f%%%19.3f%%%17ss" % (k, rmse * 100, mae * 100,(datetime.datetime.now() - starttime).seconds))
        #print("%3d%19.3f%%%19.3f" % (k, rmse , mae ))
    plt.figure(1)
    plt.subplot(121)
    plt.plot(ran, rmse0, 'b-.')
    plt.title("User-Item BasedCF-RMSE")
    plt.xlabel("Number of K")
    plt.subplot(122)
    plt.plot(ran, mae0, 'go-')
    plt.title("User-Item BasedCF-MAE ")
    plt.xlabel("Number of K")
    plt.show()