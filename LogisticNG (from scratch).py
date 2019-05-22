import getch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
import scipy.optimize as op


def plotData(datum, plot_x=[], plot_y=[], draw_line=False):
    X = datum.pop('X')  # taking the X column only
    Y = datum.pop('Y')  # taking the Y column only
    Z = datum.pop('Z')  # taking the Z column only

    for a, b, i in zip(X, Y, Z):
        if i is 1:
            plt.scatter(a, b, marker='x', color='red')      # seperating as red if value is 1
        else:
            plt.scatter(a, b, marker='x', color='blue')     # seperating as blue if value is 0

    if draw_line is True:                                   # to draw decision boundary
        plt.plot(plot_x, plot_y, 'ro-')

    plt.show()


def sigmoid(b):
    g = 1 / (1 + np.exp(-b))

    return g


def costFunction(initial_theta, X, Y):
    m = len(Y)
    J = 0
    for a, b in zip(Y, np.dot(X, initial_theta)):
        J += -a * np.log(sigmoid(b)) - (1 - a) * np.log(1 - sigmoid(b))

    J /= m

    in_mat = []

    for a, b in zip(np.dot(X, initial_theta), Y):
        in_mat.append(sigmoid(a) - b)

    in_mat = np.array(in_mat)

    size_X = X.shape
    in_mat = npm.repmat(in_mat, 1, size_X[1])

    matsum = []
    for x, y in zip(X, in_mat):
        matsum.append(x * y)

    return J, sum(matsum) / m


def decorated_cost(theta, X, Y):
    J, grad = costFunction(theta, X, Y)
    print("J is %f  " % J)
    return J


def decorated_grad(theta, X, Y):
    J, grad = costFunction(theta, X, Y)
    print("J is %f  " % J)
    return grad


def predict(theta, X):
    p = []
    for x in np.dot(X, theta):
        if x >= 0.5:
            p.append(1)
        else:
            p.append(0)

    return p


## ========================== Part 1: Plotting ================================

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
data = pd.read_csv("datang2.csv")  # For getting data
df = pd.DataFrame(data)  # Converting it to dataframe

plotData(data)

print('\nProgram paused. Press enter to continue.\n')
getch.getch()

## ========================== Part 2: Compute Cost and Gradient ===============

data = pd.read_csv("datang2.csv")  # For getting data
df = pd.DataFrame(data)
Y = df.pop('Z')
Y = np.reshape(Y, (-1, 1))

X = np.array(df)
X = np.insert(X, 0, values=1, axis=1)

initial_theta = np.zeros((X.shape[1], 1))

cost, grad = costFunction(initial_theta, X, Y)

print('Cost at initial theta (zeros): %f\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = [[-24.0], [0.2], [0.2]]
cost, grad = costFunction(test_theta, X, Y)

print('\nCost at test theta: %f\n' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

print('\nProgram paused. Press enter to continue.\n')
getch.getch()

## ============= Part 3: Optimizing using fminunc  =============

[theta, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = op.fmin_bfgs(f=decorated_cost, x0=initial_theta,
                                                                          maxiter=400, fprime=decorated_grad,
                                                                          args=(X, Y), full_output=True)

print('Cost at theta found by fminunc: %f\n' % fopt)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# plotDecisionBoundary(theta, X, y) done at the end

print('\nProgram paused. Press enter to continue.\n')
getch.getch()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

theta = np.reshape(theta, (3, -1))
prob = sigmoid(np.dot([1, 45, 85], theta))

print('For a student with scores 45 and 85, we predict an admission probability of %f\n' % prob)
print('Expected value: 0.775 +/- 0.002\n\n')

p = predict(theta, X)
accuracy = 0
for x, y in zip(p, Y):
    if x == y:
        accuracy += 1

print('Train Accuracy: %f' % accuracy)
print('Expected accuracy (approx): 89.0\n')

# 2 points required to plot line
plot_x = [min(X[:, 1]) - 2, max(X[:, 1]) + 2]
plot_y = np.multiply((-1 / theta[2]), (np.multiply(theta[1], plot_x) + theta[0]))

data = pd.read_csv("datang2.csv")  # For getting data
df = pd.DataFrame(data)
plotData(data, plot_x, plot_y, True)
