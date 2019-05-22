# Linear Regression

This is Python implementation of the Linear Regression code written in Matlab/Octave in exercise files of Machine Learning course by Andrew Ng. It contains first three parts of the code till Gradient Descent algorithm implementation.

```python
import getch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

- The first library is `getch` which is used to implement *press any key to continue* in python. The function used is `getch.getch()`
- The second library used is our favorite `pandas` library for dataframes and importing *big data*
- Third one is `matplotlib` as you can see. It is used for visualization of the data by plotting it in form of graphs
- The last one is `numpy` for handling matrices and matrix opertaions

The first section in the machine-learning-ex1 was the `warmupExercise` where we were required to generate a 5X5 matrix with 1s in it diagonal

## warmUpExercise

#### Python implementation

```python
def warmUpExercise():
    A = np.eye(5)

    return A
```
Here `numpy` has in-built function `eye()` which returns the matrix according to the value of para,eter passed. The result is stored in variable and returned.

#### Matlab/Octave implementation

```matlab
function A = warmUpExercise()
  A = [];
  A=eye(5);
end
```

## Plotting Data
The second function in the file is `plotData()` it takes the data, theta and `bool` as input aand displays the graph. 

#### Python implementation
```python
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
```
It returns the colums of the data provided by taking in the data.
- If we only have to plot data then we can simply use the first arguement and leave the rest blank. `plotData(datum)`
- If we want to draw the line after computing cost function then we can use the rest of the arguements, i.e., theta and the `bool` value e.g. `plotData(data, np.array([[1],[2]]), True)` 
- The `scatter()` function plot *scatter plot* afer taking the required arguements.
- According to the `bool` value the `if` statement is executed and the line is drawn using `X` and the `theta`
- The plot is made and the columns are returned

#### Matlab/Octave implementation
```matlab
function plotData(x, y)

figure;
  plot(x, y, "rx", "MarkerSize", 10);
  axis([4 24 -5 25]);
  xlabel("Population of City in 10,000s"); 
  ylabel("Profit in $10,000s");
end
```
