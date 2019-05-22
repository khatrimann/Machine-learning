A1 = dict()
A2 = dict()
mat = list()
res = dict()

n1 = int(input("Total elements in A1: "))

print("\nEnter Elements value and membership value of set A1, seperated by spaces: \n")

for _ in range(n1):
    n, k = input().split()
    n = int(n)
    k = float(k)
    A1[n] = k

print(A1)

n2 = int(input("Total elements in A2: "))

print("\nEnter Elements value and membership value of set A2, seperated by spaces: \n")

for _ in range(n2):
    n, k = input().split()
    n = int(n)
    k = float(k)
    A2[n] = k

print(A2)

for k1 in A1:
    p = list()
    for k2 in A2:
        x = (k1 + k2) % 3
        p.append(x)
    mat.append(p)

print(mat)

distinct = set(x for l in mat for x in l)
distinct = list(distinct)

print(distinct)

for i in range(len(distinct)):
    exec("list" + str(distinct[i]) + " = list()")
    print("created list %d" % (i + 1))

for k1 in A1:
    for k2 in A2:
        x = (k1 + k2) % 3
        # print("list" + str(x) + ".append(" + A1[k] + ")")
        exec("list" + str(x) + ".append(min([" + str(A1[k1]) + "," + str(A2[k2]) + "]))")
        # exec("list" + str(x) + ".append(" + str(A2[k2]) + ")")

for i in range(len(distinct)):
    exec("var" + str(distinct[i]) + " = max(list" + str(distinct[i]) + ")")
    exec("print(var" + str(distinct[i]) + ")")
    exec("res[" + str(distinct[i]) + "] = var" + str(distinct[i]))

print(res)

for key in res:
    print("The element is " + str(key) + " and the membership value is " + str(res[key]))
