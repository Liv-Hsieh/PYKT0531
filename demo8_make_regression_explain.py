import matplotlib.pyplot as plt
from sklearn import datasets
from pprint import pprint
from numpy import array

data1 = datasets.make_regression(10, 6, noise=5)
for i in range(0, 6):
    x = data1[0][:, i]
    y = data1[1]
    plt.scatter(x, y)
    #plt.show()

ro = array(sorted(data1[0], key=lambda t: t[0]))
r1 = array(sorted(data1[0], key=lambda t: t[1]))
r2 = array(sorted(data1[0], key=lambda t: t[2]))
r3 = array(sorted(data1[0], key=lambda t: t[3]))
r4 = array(sorted(data1[0], key=lambda t: t[4]))
r5 = array(sorted(data1[0], key=lambda t: t[5]))

print("now sort by second column(r1)")
pprint(r1)
print("now sort by third column(r2)")
pprint(r2)
