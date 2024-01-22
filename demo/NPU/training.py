import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_x = []
training_y = []
for x in range(10000):
    a = np.random.randint(2,size=16)
    b = np.random.randint(2,size=16)
    a = np.reshape(a, (4,4))
    b = np.reshape(b, (4,4))
#    print(a)
#    print(b)
    c = np.matmul(a,b)
    c = np.where(c>1, 1, 0)
#    print(c)
    a= np.reshape(a, (16))
    b= np.reshape(b, (16))
    c= np.reshape(c, (16))
    training_x = np.append(training_x,[a,b])
#    print(training_x)
    training_y = np.append(training_y,[c])
#    print(training_y)
training_x = np.reshape(training_x,(10000,32))
#print(training_x)
training_y = np.reshape(training_y,(10000,16))
#print(training_y)