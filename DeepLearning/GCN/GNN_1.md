# Graph Neural Network

``` python
import numpy as np

# adjacency matrix
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]], dtype=float)

# node feature
X = np.matrix([
    [i, -i]
    for i in range(A.shape[0])], dtype=float)

# print(A * X)

# identity matrix
I = np.matrix(np.eye(A.shape[0]))

A_hat = A + I

# print(A_hat * X)

# degree matrix
D = np.array(np.sum(A, axis=1))
D = [x[0] for x in D]
D = np.matrix(np.diag(D))

# print(D ** -1 * A * X)

# weight
W = np.matrix([[1, -1],
               [-1, 1]])

D_hat = np.array(np.sum(A_hat, axis=1))
D_hat = [x[0] for x in D_hat]
D_hat = np.matrix(np.diag(D_hat))

print(D_hat ** -1 * A_hat * X * W)

# add activation function, such as relu
```
