import numpy as np
from collections import OrderedDict

np.set_printoptions(threshold=np.nan)

# trainning data as (2707, 256)-array
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")

# label values
train_out = np.genfromtxt("data/train_out.csv")

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

print np.around(radii, decimals = 2)

# find the
distancematrix = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        # print "Distance (" + str(i) + ", " + str(j) + ") = " + str(np.linalg.norm(centers[i,:]-centers[j,:]))
        distancematrix[i,j] = np.around((np.linalg.norm(centers[i,:]-centers[j,:])), decimals=2)

for i in range(10):
    print np.around(np.mean(distancematrix[i,:]), decimals =2)

# print tables
# for i in range(10):
#     print "\\textbf{" + str(i)+ "}&"
#
# for i in range(10):
#     print "\\textbf{" + str(i)+ "}&" + '&'.join(str(p) for p in list(distancematrix[i,:])) + "\\\\"
#     print "\\hline"
