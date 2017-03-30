import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.set_printoptions(threshold=np.nan)

# load data
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

"""
task 4
"""
#for task 4 we are going to build a single layer network with 256+1 inputs and 10 output neurons
#the network select the max output neuron as the prediction digit
# the linear function used here yi = w0x0 + w1x1 + .... + w255x255 + w256(bias) for i of each digits

#first we add x256 which to times with the bias to the inputs, intert it to the end of the array
train_in = np.insert(train_in, 256, 1, axis=1)
test_in = np.insert(test_in, 256, 1, axis=1)

#initial
sum_temp = np.zeros(10)

iterations = 40
runs = 10

mis_train_all = np.zeros((runs, iterations))
mis_test_all = np.zeros((runs, iterations))

# number of misclassfied digits for each run
errors = np.zeros(10)

# interation of the trainning
for run in range(runs):
    weights = np.random.random((10,257))
    mis_train = np.zeros(iterations)
    mis_test = np.zeros(iterations)
    mis_min = len(test_in)
    opt_iteration = 0
    for i in range(iterations):
        # decrease the learning rates over time
        learning_rate = 9**(float(-i)/20 +1)
        #####################################
        # training:
        for j in range(len(train_in)):
            for k in range(10):
                sum_temp[k] = np.sum(np.multiply(weights[k], train_in[j]))
            # if theres a misclassified digit we need to update the weights
            if train_out[j] != np.argmax(sum_temp):
                #increase the weights of target digit
                weights[int(train_out[j])] += np.multiply(train_in[j], learning_rate)
                #decrease the weights of the digit being misclassfied as
                weights[np.argmax(sum_temp)] -= np.multiply(train_in[j], learning_rate)
                mis_train[i] += 1
        #####################################

        # here we use the testing data to avoid the overfitting
        for l in range(len(test_in)):
            for k in range(10):
                sum_temp[k] = np.sum(np.multiply(weights[k], test_in[l]))
            # if theres a misclassified digit
            if test_out[l] != np.argmax(sum_temp):
                mis_test[i] += 1

        # ratchet: save the best mistake rate and weights configuration seen so far
        if mis_test[i] < mis_min:
            opt_weights = np.copy(weights)
            opt_iteration = i
            mis_min = mis_test[i]

    # mistake rates
    mis_train = mis_train/len(train_in)
    mis_test = mis_test/len(test_in)

    errors[run] = mis_test[opt_iteration]
    mis_test_all[run] = mis_test
    mis_train_all[run] = mis_train

    print ("THE BEST ITERATION IS " + str(opt_iteration + 1))
    print ("WITH THE BEST MISTAKE RATE of " + str(mis_test[opt_iteration]))

print ("THE AVERAGE ACCURACY IS " + str(1 - np.mean(errors)))

print np.mean(mis_train_all, axis=0), np.std(mis_train_all, axis=0)
print np.mean(mis_test_all, axis=0), np.std(mis_test_all, axis=0)

iter_x = np.linspace(0,iterations,iterations)
plt.figure()
axes = plt.gca()
axes.set_ylim([0,0.25])
plt.errorbar(iter_x, np.mean(mis_train_all, axis=0), yerr=np.std(mis_train_all, axis=0) , label = "training mistake rate")
plt.errorbar(iter_x, np.mean(mis_test_all, axis=0), yerr =np.std(mis_test_all, axis=0) , label = "test mistake rate")
plt.legend(loc=0)
plt.show()
