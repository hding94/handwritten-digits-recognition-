import numpy as np
from collections import OrderedDict

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.set_printoptions(threshold=np.nan)

# trainning data as (1707, 256)-array
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")


# create ordered dictionary with (list_#digit: vectors)
d = OrderedDict(("list_" + str(i), []) for i in range(10))
for i in range(10):
    for j in range(len(train_out)):
        if train_out[j] == i:
            d["list_" + str(i)].append(train_in[j,:])

# means/centers as (10,256)-array
centers = np.zeros((10,256))
for i in range(10):
    centers[i,:] = np.mean(d["list_" + str(i)], axis=0)

# number of points that belong to C_i, n_i
for i in range(10):
    print "Number of " + str(i) + "s: " + str(len(d["list_" + str(i)]))

# calculate radii
radii = np.zeros((10,1))
for i in range(10):
    radius = 0
    for point in d["list_" + str(i)]:
        newradius = np.linalg.norm(point-centers[i,:])
        if newradius >= radius:
            radius = newradius
    radii[i] = radius

# create a distance matrix between centers
centers_dist = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        centers_dist[i,j] = np.linalg.norm(centers[i,:]-centers[j,:])

print("Distances between each centers")
print centers_dist

##########################################################################
"""
task 2
"""

#classifier: put the testing digits into the grou-p which closest to its center

#create confusion matrix for train data
train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))
for i in range(len(train_in)):
    #use current_dist to store the distance matrix of point i to each center
    current_dist = pairwise_distances(centers, train_in[i], metric='cosine')
    #store the shortest distance index to the prediction array
    train_pre[i] = np.argmin(current_dist)

conf_matrix_train = confusion_matrix(train_out, train_pre)

#calcute the correctly classified digits
correct_rate_train = np.zeros(10)
for i in range(10):
    correct_rate_train[i] = float(conf_matrix_train[i,i])/np.sum(conf_matrix_train[i,:])
print("correct rate of training data")
print correct_rate_train

#create confusion matrix for test data
for i in range(len(test_out)):
    current_dist = pairwise_distances(centers, test_in[i], metric='cosine')
    test_pre[i] = np.argmin(current_dist)

conf_matrix_test = confusion_matrix(test_out, test_pre)

correct_rate_test = np.zeros(10)
for i in range(10):
    correct_rate_test[i] = float(conf_matrix_test[i,i])/np.sum(conf_matrix_test[i,:])
print("correct rate of testing data")
print correct_rate_test

##########################################################
# plot confusion matrix using the function from the plot_confusion_matrix example:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
########################################################################

np.set_printoptions(precision=2)
#plot confusion martix:

#train confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix_train, classes=class_name, title = 'Confusion matix training set')
plt.savefig("traincosine.png")

#test confusion martix
plt.figure()
plot_confusion_matrix(conf_matrix_test, classes=class_name, title='Confusion matrix test set')
plt.savefig("testcosine.png")
