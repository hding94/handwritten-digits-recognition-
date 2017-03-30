import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.set_printoptions(threshold=np.nan)

# training data as (1707, 256)-array
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

# create ordered dictionary to categorize the digits with (list_#digit: vectors)
digits = OrderedDict(("list_" + str(i), []) for i in range(10))
addressDict = OrderedDict(("list_" + str(i), []) for i in range(10))
digits_test = OrderedDict(("list_" + str(i), []) for i in range(10))
addressDict_test = OrderedDict(("list_" + str(i), []) for i in range(10))
for i in range(10):
    for j in range(len(train_out)):
        if train_out[j] == i:
            digits["list_" + str(i)].append(train_in[j,:])
            addressDict["list_" + str(i)].append(j)
    for k in range(len(test_out)):
        if test_out[k] == i:
            digits_test["list_" + str(i)].append(test_in[k,:])
            addressDict_test["list_" + str(i)].append(k)

# means/centers as (10,256)-array
centers = np.zeros((10,256))
for i in range(10):
    centers[i,:] = np.mean(digits["list_" + str(i)], axis=0)

# number of points that belong to C_i, n_i
for i in range(10):
    print "Number of " + str(i) + "s: " + str(len(digits["list_" + str(i)]))

# calculate radii
radii = np.zeros((10,1))
for i in range(10):
    radius = 0
    for point in digits["list_" + str(i)]:
        newradius = np.linalg.norm(point-centers[i,:])
        if newradius >= radius:
            radius = newradius
    radii[i] = radius

# create a distance matrix between centers
centers_dist = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        centers_dist[i,j] = pairwise_distances(centers[i,:], centers[j,:], metric='euclidean')

##########################################################################
"""
task 2
"""

# classifier uses distance to center as feature:
# put the testing digits into the group which is closest to its center

# create confusion matrix for train data
train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))
for i in range(len(train_in)):
    # use current_dist to store the distance matrix of point i to each center
    current_dist = pairwise_distances(centers, train_in[i], metric='euclidean')
    # store the shortest distance index to the prediction array
    train_pre[i] = np.argmin(current_dist)

conf_matrix_train = confusion_matrix(train_out, train_pre)

# calcute the correctly classified digits
correct_rate_train = np.zeros(10)
for i in range(10):
    correct_rate_train[i] = float(conf_matrix_train[i,i])/np.sum(conf_matrix_train[i,:])
print("correct rate of training data")
print correct_rate_train

# create confusion matrix for test data
for i in range(len(test_out)):
    current_dist = pairwise_distances(centers, test_in[i], metric='euclidean')
    test_pre[i] = np.argmin(current_dist)

conf_matrix_test = confusion_matrix(test_out, test_pre)

correct_rate_test = np.zeros(10)
for i in range(10):
    correct_rate_test[i] = float(conf_matrix_test[i,i])/np.sum(conf_matrix_test[i,:])
print("correct rate of testing data")
print correct_rate_test

##########################################################
# plot confusion matrix using the function from the plot_confusion_matrix
# example:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_name = np.array([0,1,2,3,4,5,6,7,8,9])
np.set_printoptions(precision=2)

#train confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix_train, classes=class_name, title = 'train confusion matix')
plt.show()

#test confusion martix
plt.figure()
plot_confusion_matrix(conf_matrix_test, classes=class_name, title='test confusion matrix')
plt.show()
############################################################################################
"""
task 3
"""
# take two categories of the digits from dictionary digits
# here we chose the 0 and 6 as they are most likely to be misclassified in the previous task

digits0 = np.array(digits["list_0"])
test0 = np.array(digits_test["list_0"])

digits6 = np.array(digits["list_6"])
test6 = np.array(digits_test["list_6"])

# Here for the two classes classifier we use the function y(x) = dis1(x) - dis2(x), if f(x)<0 it belongs to class 1, else class 2
# def bayes_function(digist, center1, center2):
dist0 = np.zeros(len(digits0))
dist6 = np.zeros(len(digits6))
dist_test0 = np.zeros(len(test0))
dist_test6 = np.zeros(len(test6))
mis_counts0 = np.zeros(80)
mis_counts6 = np.zeros(80)
mis_test0 = 0.0
mis_test6 = 0.0

for i in range(len(dist0)):
    dist0[i] = pairwise_distances(digits0[i], centers[0], metric="euclidean") - pairwise_distances(digits0[i], centers[6], metric="euclidean")
    for j in range(80):
        if dist0[i] > -4 + 0.1*j:
            mis_counts0[j] += 1

for i in range(len(dist6)):
    dist6[i] = pairwise_distances(digits0[i], centers[0], metric="euclidean") - pairwise_distances(digits6[i], centers[6], metric="euclidean")
    for j in range(80):
        if dist6[i] <= -4 + 0.1*j:
            mis_counts6[j] += 1

misclassified = mis_counts0 + mis_counts6
mis_rate0 = np.copy(mis_counts0)/len(dist0)
mis_rate6 = np.copy(mis_counts6)/len(dist6)
mis_rate = np.copy(misclassified)/(len(dist0) + len(dist6))
#plot the histogram
bins = np.linspace(min(dist0), max(dist6), 60)
plt.hist(dist0, bins, alpha=0.5, label='digit0')
plt.hist(dist6, bins, alpha=0.5, label='digit6')
plt.legend(loc='upper right')
plt.show()

# plot the misclassfied rate according to the classifer position
mis_x = np.linspace(-4,4,80)
opt_classifier = -4 + 0.1*np.argmin(misclassified)
plt.figure(3)
plt.plot(mis_x,mis_rate0, label = "digit0 misclassified rate")
plt.plot(mis_x,mis_rate6, label = "digit6 misclassified rare")
plt.plot(mis_x,mis_rate, label = "total misclassified rate")
plt.legend(loc=0)
plt.show()

print ("THE OPTIMAL CLASSIFIER WOULD BE AT " + str(opt_classifier))
print ("IN TRAINNING SET THE MISCALSSIFIED RATE IS " + str(min(misclassified)/(len(dist0)+len(dist6))))
print "AMONG WHICH THE MISCLASSFIED RATE OF DIGIT 0 IS " + str(mis_counts6[np.argmin(misclassified)]/len(dist0))
print "AMONG WHICH THE MISCLASSFIED RATE OF DIGIT 6 IS " + str(mis_counts0[np.argmin(misclassified)]/len(dist6))
print "THE MISCLASSFIED DIGITS 0 AND 6 ARE " + str(mis_counts0[np.argmin(misclassified)]) +" " + str(mis_counts6[np.argmin(misclassified)])

##### apply the classfier on test dataset
for i in range(len(test0)):
    dist_test0[i] = pairwise_distances(test0[i], centers[0], metric="euclidean") - pairwise_distances(test0[i], centers[6], metric="euclidean")
    if dist_test0[i] > opt_classifier:
        mis_test0 += 1
for i in range(len(test6)):
    dist_test6[i] = pairwise_distances(test6[i], centers[0], metric="euclidean") - pairwise_distances(test6[i], centers[6], metric="euclidean")
    if dist_test6[i] <= opt_classifier:
        mis_test6 += 1
print
print "IN TEST SET THE MISCLASSFIED RATE OF DIGIT 0 IS " + str(mis_test6/len(test0))
print "IN TEST SET THE MISCLASSFIED RATE OF DIGIT 6 IS " + str(mis_test0/len(test6))
print "IN TEST SET THE OVERALL MISCLASSFIED RATE IS " + str((mis_test0 + mis_test6)/(len(test6)+len(test0)))
