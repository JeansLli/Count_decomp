import numpy as np
import matplotlib.pyplot as plt

from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti

def creat_random_points(x_range,y_range,N):
    # output is a (N,2) array
    points = np.random.rand(N,2)
    points[:,0]=points[:,0] * x_range
    points[:,1]=points[:,1] * y_range
    return points

# Create the data
points = creat_random_points(10,10,10)

# Plot the data
#plt.figure(figsize=(10,10))
#plt.scatter(points[:,0], points[:,1], alpha=0.5)
#plt.show()



# Compute the hilbert function
ss = [0, 1, 2, 3, 4, 5]
ks = [1, 3 / 4, 1 / 2, 1 / 4, 0]

p = Persistable(points)
bf = p._bifiltration
hf = bf._hilbert_function(ss, ks, reduced=False, n_jobs=4)
print("hilbert function=\n", hf)

# Compute the signed Betti barcode
sb = signed_betti(hf)
print("signed Betti barcode=\n",sb)

check = np.zeros(hf.shape)
for i in range(hf.shape[0]):
    for j in range(hf.shape[1]):
        for i_ in range(0, i + 1):
            for j_ in range(0, j + 1):
                check[i, j] += sb[i_, j_]

print(np.testing.assert_equal(check, hf))
print("check=",check)


num_pos_bar = 0 
num_neg_bar = 0
num_bars = num_pos_bar + num_neg_bar
print("number of positive bars in the signed Betti barcode=",num_pos_bar)
print("number of negative bars in the signed Betti barcode=",num_neg_bar)
print("number of bars in the signed Betti barcode=",num_bars)

# Compute the maximal simplicial complex




