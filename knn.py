#John Kaufman
#Python implementation of k-nearest-neighbors machine learning algorithm
#
#     This code will read in training and test data from two csv files, and for each test point, it will find the
#   k training points that have the closest euclidian distance to the test point. The test point will be assigned the class
#   that the majority of the k-nearest-neighbors have. ()


import numpy as np

#point_distance function:
# Find the distance between a test point and each training point
#   test_point: a single test point, represented as a 1d numpy array
#   train_set: the training data, represented as a 2d numpy array
#   Returns distance, a 2d numpy array that shows the distance from the points
def point_distance(test_point, train_set):

    difference =  train_set - test_point

    distance = np.linalg.norm(difference, axis=1)

    return distance



#doknn function:
# Perform the knn algorithm
#   train: the training data from the train.csv file
#   test: the test data from the test.csv file
#   k: the number of neighbors to consider when training the model
#   Return results, the classification for each test point
def doknn(train, test, k):


    #Initialize arrays to store the distances
    distances = np.zeros((test.shape[0], train.shape[0]))
    sorted = distances;

    for i in range(0, test.shape[0]): #for each data point in the test set

        #Message to make sure the loop is working
        if i%200 == 0:
            print("Analyzing test point: ", i)

        #Find the distance from the test point to each training point
        distances[i] = point_distance(test[i,1:train.shape[1]-1],train[:,1:train.shape[1]-1])


    #Distances is now a full, accurate matrix
    
    #Sort the distances array to find the nearest neighbors
    for i in range(0, test.shape[0]):
        sorted[i] = np.argsort(distances[i])

    #running_sum tracks the class belonging to the most neighbors
    running_sum = 0
    
    #Results will contain each test points class
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

        #if running_sum is greater than 0, then there were more 1s, and the test point is assigned 1.
        if running_sum > 0:
            results[int(i)][1] = 1
        elif running_sum < 0:
            results[int(i)][1] = 0


    return results


#cross_val function
# Perform K-fold cross validation on the training set
#   train: the training data
#   K: the number of folds to use
def cross_val(train, K):

    #define an interval based on K and the size of the training set
    interval = int(train.shape[0]/K)

    #Specify a subset of the training set to train on
    sub_train = np.zeros(((K-1)*interval, train.shape[1]))

    #Specify the subset to validate on
    validation = np.zeros((interval, train.shape[1]))

    accuracies = np.zeros(K)

    #Go through each fold, validating on each one
    for i in range(K):

        #Fill in the subsets
        for j in range(K):
            #This subset is the validation set
            if j == i:
                validation = train[(j*interval):((j+1)*interval)]
            
            #The other subsets are training sets for validation
            elif j < i:
                sub_train[(j*interval):((j+1)*interval)] = train[(j*interval):((j+1)*interval)]
            else:
                sub_train[((j-1)*interval):(j*interval)] = train[(j*interval):((j+1)*interval)]

        #Perform cross validation on the current validation set
        #Plug the result directly into the accuracy function to check the accuracy
        accuracies[i] = accuracy(doknn(sub_train, validation, 3)[:,1], validation[:,train.shape[1]-1])
        print("For validation set ", i, "accuracy was ", accuracies[i])

#accuracy function
# Determine the accuracy of each fold's validation
#   results: the classification results from the knn algorithm
#   true_values: the true classes from the validation data
#   returns the percentage of accurate classifications
def accuracy(results, true_values):

    num_correct = 0

    for i in range(results.size):
        if results[i] == true_values[i]:
            num_correct+=1

    return (num_correct/results.size)

def main():

    #Get the training and test data from csv files
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


