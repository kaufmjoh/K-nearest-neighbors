import numpy as np
import math

#calculate the distance between test point a and train point b
def point_distance(a, b):

    #print("Measuring the distance between ", a)
    #print("and ", b)

    distance = 0

    difference =  np.subtract(a,b)

    #print(difference)

    distance = np.linalg.norm(difference)

    #print(distance)

    return distance



def doknn(train, test, k):

    print("Performing knn on train, which has ", train.shape[0], " members with ", train.shape[1], "attributes")
    print("and on test, which has ", test.shape[0], " members with ", test.shape[1], "attributes")


    #print(np.matrix(train))

    distances = np.zeros((test.shape[0], train.shape[0]))
    sorted = distances;

    for i in range(0, test.shape[0]): #for each data point in test

        if i%50 == 0:
            print("Analyzing: ", i)

        for j in range(0, train.shape[0]): #for each training data point
            distances[i][j] = point_distance(test[i,1:train.shape[1]-1],train[j,1:train.shape[1]-1])#measure the distance between two points


    print (np.matrix(distances))

    #distances is now a full, accurate matrix
    for i in range(0, test.shape[0]):
        sorted[i] = np.argsort(distances[i])


    print(np.matrix(sorted))

    running_sum = 0
    results = np.zeros((test.shape[0], 2))

    for i in range(0, test.shape[0]): #for each test point
        running_sum = 0
        results[i][0] = i
        for j in range(0, k): #for each neighbor

            print(sorted[i][j])

            if train[int(sorted[i][j])][((train.shape[1])-1)] == 1:
                running_sum+=1
            else:
                running_sum-=1

        if running_sum > 0:
            results[i][1] = 1
        else:
            results[i][1] = 0


    return results

def main():

    print("General kenobi")
    train=np.genfromtxt('./train.csv',delimiter=',',skip_header=1)
    test=np.genfromtxt('./test_pub.csv',delimiter=',',skip_header=1)


    #for debuggin purposes, only use first 1000 rows of train
    nano_train = train[0:12]

    #knn = doknn(train, test, 11)

    cross_val(nano_train, 4)

def cross_val(train, K):
    #Capital K-fold Cross Validation

    interval = int(train.shape[0]/K)

    sub_train = np.zeros(((K-1)*interval, train.shape[1]))
    validation = np.zeros((interval, train.shape[1]))

    #print(np.matrix(train))
    #print("train--------------------------------------------------")

    #validation = [[0]*train.shape[1]]*(train.shape[0]/K)

    accuracies = np.zeros(K)

    for i in range(K):

        print("validation set is: ", i)

        for j in range(K):
            #print(j)
            #print("index (0-3)$$$$$$$$$$$$$$$$$")

            if j == i:
                validation = train[(j*interval):((j+1)*interval)]
            elif j < i:
                #print(train[(j*interval):((j+1)*interval)])
                #print("train subset*****************************************************")
                sub_train[(j*interval):((j+1)*interval)] = train[(j*interval):((j+1)*interval)]
            else:


                #print(train[(j*interval):((j+1)*interval)])
                #print("train subset*****************************************************")
                sub_train[((j-1)*interval):(j*interval)] = train[(j*interval):((j+1)*interval)]

            #print(validation[1:100,train.shape[1]-1])
       # print(np.matrix(validation))
       # print("validation set--------------------------------------------------")
       # print(np.matrix(sub_train))
       # print("sub_train set--------------------------------------------------")

#
        accuracies[i] = accuracy(doknn(sub_train, validation, 3)[:,1], validation[:,train.shape[1]-1])
        print("For validation set ", i, "accuracy was ", accuracies[i])


def accuracy(results, true_values):
    num_correct = 0

    print(np.matrix(results))
    print(np.matrix(true_values))

    for i in range(results.size):
        if results[i] == true_values[i]:
            num_correct+=1


    return (num_correct/results.size)

main()


