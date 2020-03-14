import os
import sys
from mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

print(x_train.shape)
# (60000, 784)
print(t_train.shape)
# (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(np.random.choice(60000, 10))
# [24598 13497 47089 19298 35610 21929 59697 46776  8818 40623]
