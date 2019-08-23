import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


filePath = 'test.csv'
df = pd.read_csv(filePath)

my_data = np.genfromtxt('test.csv', delimiter=',')  # Read the data

x = my_data[:, 0].reshape(-1, 1)  # -1 tells numpy to figure out the dimension by itself
y = my_data[:, 1].reshape(-1, 1)  # create y matrix
plt.scatter(my_data[:, 0].reshape(-1, 1), y)
#  plt.show()

# Declaring learning rate and number of iteration.
# Small alpha means slow learning rate.
alpha = 0.0001
iters = 1000
theta = np.array([[1, 1]])


def computecost(x, y, theta):
    # @ means matrix multiplication of arrays. If we want to use *
    # for multiplication we will have to convert all arrays to matrices
    inner = np.power(((x @ theta) - y), 2)

    return np.sum(inner) / (2 * len(x))


def gradientDescent(x, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(x)) * np.sum((x @ theta - y) * x, axis=0)
        cost = computecost(x, y, theta)
        #  if i % 10 == 0: # just look at cost every ten loops for debugging
        #   print(cost)
    return (theta, cost)


g, cost = gradientDescent(x, y, theta, alpha, iters)
print(g, cost)

plt.scatter(my_data[:, 0].reshape(-1, 1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals #  the line equation
plt.plot(x_vals, y_vals, '--')

plt.show()
