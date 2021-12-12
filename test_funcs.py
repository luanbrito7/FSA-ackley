import numpy as np
import random
import math

###### Single-peak functions ######

def f1(X):
    return sum(np.square(X))

def f2(X):
    sum, prod = 0, 1
    for val in X:
        sum += np.abs(val)
        prod *= np.abs(val)
    return sum + prod

def f3(X):
    sum, temp = 0, 0
    for i in range(len(X)):
        for j in range(i):
            temp += X[j]
        sum += temp**2
    return sum

def f4(X):
    return np.maX(np.abs(X))

def f5(X):
    sum = 0
    for i in range(len(X) - 1):
        sum += 100 * (X[i+1] - X[i]**2)**2 + (X[i] - 1)**2
    return sum

def f6(X):
    sum = 0
    for val in X:
        sum += (0.5 + val)**2
    return sum

def f7(X):
    sum = 0
    for i in range(len(X)):
        sum += i * X[i]**4
    return sum + random.random()

def f8(X):
    sum = 0
    for i in range(len(X)):
        sum += pow(np.abs(X[i]), i+1)
    return sum

def F9(X):
    sum = 0
    for i in range(len(X)):
        sum += i * X[i]**2
    return sum


###### Multi peak functions #######

def f10(X):
    sum = 0
    for i in range(len(X)):
        sum+= -X[i] * math.sin(math.sqrt(abs(X[i])))
    return sum

def f11(X):
    return np.sum(X*X - 10*np.cos(2*np.pi*X)) + (10*np.size(X))

def f12(X):
    s1=0
    s2=0
    for i in range(len(X)):
        s1 += X[i]**2
        s2 += math.cos(2 * math.pi * X[i])
    a = -20*math.exp(-0.2*math.sqrt(s1/len(X)))
    b = math.exp(s2/len(X)) + 20 + math.exp(1)
    return a + b

def f13(X):
    s = sum(np.asarray(X) ** 2)
    p = 1
    for i in range(len(X)):
        p *= math.cos(X[i] / math.sqrt(i+1))
    return 1 + s / 4000 - p

def f14(X):
    sum = 0
    for i in range(len(X)-1):
        o = ((X[i]-1)/4)+1
        sum += pow(o-1, 2) * (1 + 10*pow(math.sin(math.pi*o+1),2))
    a = pow((X[-1]-1)/4,2) * (1+pow((math.sin(2*math.pi*(1+(X[-1]-1)/4))),2))
    return pow((math.sin((1+(X[0]-1)/4)*math.pi)),2)+sum+np.square((1+(X[-1]-1)/4)-1)

def f15(X):
    val = 0
    d = len(X)
    for i in range(d):
        val += X[i] * math.sin(math.sqrt(abs(X[i])))
    return 418.9829 * d - val

def f16(X):
    a1 = sum(np.square(X))
    a2 = np.sin(np.sqrt(a1))
    a = math.pow(a2,2) - 0.5
    b = math.sqrt(1 + 0.001 * sum(np.square(X)))
    return 0.5 + a / b


###### Fixed-dimensional functions ######

def f17(X):
    return 100*math.sqrt(abs(X[1]-0.01*X[0]**2))+0.01*abs(X[0]+10)

def f18(X):
    return -0.0001*pow((abs(math.sin(X[0])*math.sin(X[1])*math.exp(abs(100-(pow((X[0]*X[0]+X[1]*X[1]),1/2))/math.pi)))+1),0.1)

def f19(X):
    return -1*(1+math.cos(12*pow((X[1]*X[1]+X[0]*X[0]),1/2)))/(0.5*(X[0]*X[0]+X[1]*X[1])+2)

def f20(X):
    return 0.5+(pow((math.sin(X[0]*X[0]-X[1]*X[1])),2)-0.5)/(pow((1+0.001*(X[0]*X[0]+X[1]*X[1])),2))

def f21(X):
    return 0.26*(X[0]*X[0]+X[1]*X[1])-0.48*X[0]*X[1]


def f22(X):
    return 2*pow(X[0],2)-1.05*pow(X[0],4)+1/6*pow(X[0],6)+X[0]*X[1]+pow(X[1],2)

def f23(X):
    a = 100 * pow((pow(X[0], 2) - X[1]), 2)
    b = pow((X[0] - 1), 2) + pow((X[2] - 1), 2)
    c = 90 * pow((pow(X[2], 2) - X[3]), 2)
    d = 10.1 * (pow((pow(X[1], 2) - 1), 2) + pow((X[3] - 1), 2))
    f = 19.8 * (X[1] - 1) * (X[3] - 1)
    return a + b + c + d + f


###### Ackley function

def power(my_list):
  return [x**2 for x in my_list]

def cos_2_pi(my_list):
  return [math.cos(2 * math.pi * x) for x in my_list]

def ackley_function(fenotipos):
  sum_1 = sum(power(fenotipos))
  sum_2 = sum(cos_2_pi(fenotipos))

  termo_1 = -20 * math.exp(-0.2 * math.sqrt(sum_1/30))
  termo_2 = -math.exp(sum_2/30)

  return termo_1 + termo_2 + 20 + math.e