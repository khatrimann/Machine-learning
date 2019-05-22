import pandas as pd
from sklearn.preprocessing import Imputer
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

label1 = """Internal Component 1
Marks calculated as per assigned Weightage
(weightage=   2.5)"""

label2 = """Internal Component 2
Marks calculated as per assigned Weightage
(weightage=   2.5)"""

label3 = """Internal Component 3(Quiz)
Marks calculated as per assigned Weightage
(weightage=   20)"""

label4 = """Internal Component 4(Project)
Marks calculated as per assigned Weightage
(weightage=   15)"""

label5 = """Mid semester
Marks calculated as per assigned Weightage
(weightage=   20)"""


mydata = pd.read_excel("/home/mann/Downloads/imputation.xlsx")
target1 = mydata[label1]
target2 = mydata[label2]
target3 = mydata[label3]
target4 = mydata[label4]
target5 = mydata[label5]
names = mydata["First name"]

xs = np.arange(len(names))
target1mask = np.isfinite(target1)
target2mask = np.isfinite(target2)
target3mask = np.isfinite(target3)
target4mask = np.isfinite(target4)
target5mask = np.isfinite(target5)

fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(xs[target1mask], target1[target1mask], linestyle='-', marker='o', c='red')
plt.plot(xs[target2mask], target2[target2mask], linestyle='-', marker='o', c='blue')
plt.plot(xs[target3mask], target3[target3mask], linestyle='-', marker='o', c='green')
plt.plot(xs[target4mask], target4[target4mask], linestyle='-', marker='o', c='violet')
plt.plot(xs[target5mask], target5[target5mask], linestyle='-', marker='o', c='yellow')

red_patch = mpatches.Patch(color='red', label='Internal component 1')
blue_patch = mpatches.Patch(color='blue', label='Internal component 2')
green_patch = mpatches.Patch(color='green', label='Internal component 3')
violet_patch = mpatches.Patch(color='violet', label='Internal component 4')
yellow_patch = mpatches.Patch(color='yellow', label='Mid sem')
plt.legend(handles=[red_patch, blue_patch, green_patch, violet_patch, yellow_patch])

plt.title('Missing values')
plt.ylabel('marks')
plt.xlabel('First name')

plt.show()

target1 = mydata[label1]
target2 = mydata[label2]
target3 = mydata[label3]
target4 = mydata[label4]
target5 = mydata[label5]

print("Standard Deviation for IC1: %s" % str(np.std(target1)))
print("Standard Deviation for IC2: %s" % str(np.std(target2)))
print("Standard Deviation for IC3: %s" % str(np.std(target3)))
print("Standard Deviation for IC4: %s" % str(np.std(target4)))
print("Standard Deviation for IC5: %s" % str(np.std(target5)))
print("Mean before: %s" % str(np.mean(target1)))
print("Mean before: %s" % str(np.mean(target2)))
print("Mean before: %s" % str(np.mean(target3)))
print("Mean before: %s" % str(np.mean(target4)))
print("Mean before: %s" % str(np.mean(target5)))

target1 = target1.values.reshape(-1, 1)
target2 = target2.values.reshape(-1, 1)
target3 = target3.values.reshape(-1, 1)
target4 = target4.values.reshape(-1, 1)
target5 = target5.values.reshape(-1, 1)

print("Select a method:\n1. Mean\n2. Median\n3. Most Frequent(mode)\n")
ip = int(input())

if ip is 1:
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
elif ip is 2:
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
elif ip is 3:
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
else:
    print("Invalid choice")
    sys.exit()

target1 = imp.fit_transform(target1)
target2 = imp.fit_transform(target2)
target3 = imp.fit_transform(target3)
target4 = imp.fit_transform(target4)
target5 = imp.fit_transform(target5)

print(label1)
print(target1)

print("\n")

print(label2)
print(target2)

print("\n")

print(label3)
print(target3)

print("\n")

print(label4)
print(target4)

print("\n")

print(label5)
print(target5)

target1 = target1.ravel()
target2 = target2.ravel()
target3 = target3.ravel()
target4 = target4.ravel()
target5 = target5.ravel()

print("Standard Deviation for IC1 after imputation: %s" % str(np.std(target1)))
print("Standard Deviation for IC2 after imputation: %s" % str(np.std(target2)))
print("Standard Deviation for IC3 after imputation: %s" % str(np.std(target3)))
print("Standard Deviation for IC4 after imputation: %s" % str(np.std(target4)))
print("Standard Deviation for IC5 after imputation: %s" % str(np.std(target5)))
print("Mean after: %s" % str(np.mean(target1)))
print("Mean after: %s" % str(np.mean(target2)))
print("Mean after: %s" % str(np.mean(target3)))
print("Mean after: %s" % str(np.mean(target4)))
print("Mean after: %s" % str(np.mean(target5)))

fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(target1, c='red', marker='o')
plt.plot(target2, c='blue', marker='o')
plt.plot(target3, c='green', marker='o')
plt.plot(target4, c='violet', marker='o')
plt.plot(target5, c='yellow', marker='o')

red_patch = mpatches.Patch(color='red', label='Internal component 1')
blue_patch = mpatches.Patch(color='blue', label='Internal component 2')
green_patch = mpatches.Patch(color='green', label='Internal component 3')
violet_patch = mpatches.Patch(color='violet', label='Internal component 4')
yellow_patch = mpatches.Patch(color='yellow', label='Mid sem')
plt.legend(handles=[red_patch, blue_patch, green_patch, violet_patch, yellow_patch])

plt.title('Imputed Values')
plt.ylabel('marks')
plt.xlabel('First name')

plt.show()


plt.hist(target5, histtype='stepfilled', bins=[0, 10, 20, 30, 40, 50, 60])

plt.show()