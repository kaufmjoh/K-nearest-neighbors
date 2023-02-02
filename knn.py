#John Kaufman
#Spring 2021 CS 434: knn implementation with K-fold cross validation


#WARNING: This program only works on python3, not python2 or regular python


import numpy as np


#calculate the distance between test point a and training points b
def point_distance(test_point, train_set):

    difference =  train_set - test_point

    distance = np.linalg.norm(difference, axis=1)

    return distance



#perform knn, training on train, testing on test, with k neighbors
def doknn(train, test, k):


    distances = np.zeros((test.shape[0], train.shape[0]))
    sorted = distances;

    for i in range(0, test.shape[0]): #for each data point in test

        #Message to make sure the loop is working
        if i%200 == 0:
            print("Analyzing test point: ", i)

        distances[i] = point_distance(test[i,1:train.shape[1]-1],train[:,1:train.shape[1]-1])


    #distances is now a full, accurate matrix
    for i in range(0, test.shape[0]):
        sorted[i] = np.argsort(distances[i])


    running_sum = 0
    results = np.zeros((test.shape[0],2))

    for i in range(0, int(test.shape[0])): #for each test point
        running_sum = 0
        results[int(i)][0] = i

        for j in range(0, k): #for each neighbor


            #if the test point is 1, increment running sum, if not, decrement
            if train[int(sorted[int(i)][j])][((train.shape[1])-1)] == 1:
                running_sum+=1
            else:
                running_sum-=1

        #if running_sum is greater than 0, then there were more 1s.
        if running_sum > 0:
            results[int(i)][1] = 1
        elif running_sum < 0:
            results[int(i)][1] = 0


    return results



#perform K-fold validation on the training set
def cross_val(train, K):

    #define an interval based on K and the size of the training set
    interval = int(train.shape[0]/K)

    #a subset of the training set to train on
    sub_train = np.zeros(((K-1)*interval, train.shape[1]))

    #the subset to validate on
    validation = np.zeros((interval, train.shape[1]))

    accuracies = np.zeros(K)

    #for each fold
    for i in range(K):

        #fill the subsets
        for j in range(K):
            if j == i:
                validation = train[(j*interval):((j+1)*interval)]
            elif j < i:
                sub_train[(j*interval):((j+1)*interval)] = train[(j*interval):((j+1)*interval)]
            else:
                sub_train[((j-1)*interval):(j*interval)] = train[(j*interval):((j+1)*interval)]

        #perform cross validation on the current validation set
        #plug the result directly into the accuracy function to check the accuracy
        accuracies[i] = accuracy(doknn(sub_train, validation, 3)[:,1], validation[:,train.shape[1]-1])
        print("For validation set ", i, "accuracy was ", accuracies[i])

#determine the accuracy of a fold's cross validation
def accuracy(results, true_values):

    num_correct = 0

    for i in range(results.size):
        if results[i] == true_values[i]:
            num_correct+=1


    return (num_correct/results.size)

def main():

    train=np.genfromtxt('./train.csv',delimiter=',',skip_header=1)
    test=np.genfromtxt('./test_pub.csv',delimiter=',',skip_header=1)

    #randomize the training set for the purposes of 4-fold cross validation
    np.random.shuffle(train)


    print("Performing knn with the training and test sets:")
    test_results = doknn(train, test, 501)
    print("------------------------------------------------")

    #Write the results to a csv file
    #np.savetxt("outfile.csv",test_results,delimiter=",")

    #perform cross validation on the training set with 4 folds
    print("Performing 4-fold cross validation on the training set")
    cross_val(train, 4)


#call the driver function
main()


