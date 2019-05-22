import getch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def warmUpExercise():
    A = np.eye(5)

    return A


def plotData(datum, theta=np.array([[0], [0]]), draw_line=False):
    X = datum.pop('X')  # taking the X column only
    Y = datum.pop('Y')  # taking the Y column only

    plt.scatter(X, Y, marker='x')  # Creating a scatter plot with X and Y

    if draw_line is True:  # Adding [1]s in X at position 0
        for elem in X:
            elem = [1, elem]
            newX.append(elem)

        X = newX
        X = np.array(X)
        plt.plot(X[:, 1], np.dot(X, theta), marker='_')

    plt.show()

    return X, Y


def computeCost(X, Y, theta):
    # Initialize some useful values

    m = len(Y)  # Number of training examples

    # You need to return the following variables correctly
    J = 0
    XintoTheta = np.dot(X, theta)
    Y = np.array(Y)
    XintoThetaminusY = np.subtract(XintoTheta, Y)
    dummyArr = []
    for element in XintoThetaminusY:
        element = element * element
        dummyArr.append(element)

    J = (1 / (2 * m)) * sum(dummyArr)
    return J


def gradientDescent(X, Y, theta, alpha, iters):
    # Initialize some useful values

    m = len(Y)  # Number of training examples
    J_history = [[0]] * iters
    val = 0.0
    val1 = 0.0
    for i in range(iters):
        h = np.dot(X, theta)
        hmy = np.subtract(h, Y)
        hmyx = np.zeros([97, 1])
        i = 0
        for a, b in zip(X, hmy):
            hmyx[i] = list(a[1] * b)
            i += 1
        val = sum(hmy) / m
        val1 = sum(hmyx) / m

        temp1 = float(theta[0] - (alpha * val))
        temp2 = theta[1] - (alpha * val1)
        theta = [[temp1], temp2]

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, Y, theta)
        print()

    return theta


## ========================== Part 1: Basic Function ==========================

print("Running warmup Exercise...\n")
print("5X5 identity matrix: \n")
A = warmUpExercise()

print("Press enter to continue")
# getch.getch()

## ========================== Part 2: Plotting ================================

print("Plotting data...")
data = pd.read_csv("datang.csv")  # For getting data
df = pd.DataFrame(data)  # Converting it to dataframe

X, Y = plotData(data)  # Plotting

m = len(Y)
Y = np.reshape(Y, (97, 1))  # reshaping into columns as it will not be compatible at the tine if
                            # operations like matrix multiplication

print('Program paused. Press enter to continue.\n')
# getch.getch()

## ========================== Part 3: Cost and Gradient descent ===============

newX = []

for elem in X:              # Add a column of ones to x
    elem = [1, elem]
    newX.append(elem)

X = newX                    # Adding colums of 1s at starting of matrix X
theta = [[0], [0]]          # initialize fitting parameters

X = np.array(X)             # converting matrices to
Y = np.array(Y)             # numpy array for operations
theta = np.array(theta)

# Some gradient descent settings
iters = 1500
alpha = 0.01

# compute and display initial cost
print('\nTesting the cost function ...\n')
J = computeCost(X, Y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, Y, [[-1], [2]])
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 54.24\n')

print('Program paused. Press enter to continue.\n')
# getch.getch()

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, Y, theta, alpha, iters)

# print theta to screen
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plotData(pd.read_csv("datang.csv"), theta, True)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of')
print(predict1 * 10000)
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of')
print(predict2 * 10000)

print('Program paused. Press enter to continue.\n')
# getch.getch()
