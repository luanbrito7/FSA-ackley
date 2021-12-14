import random
import time
import numpy as np
import matplotlib.pyplot as plt
import statistics
import test_funcs as f

rng = np.random.default_rng()

def fun(X):
    return f.f11(X)

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    return X

def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

def BorderCheck(X,lb,ub,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]<lb[j]:
                X[i,j] = ub[j]
            elif X[i,j]>ub[j]:
                X[i,j] = lb[j]
    return X

def rand_1():
    a=random.random()
    if a>0.5:
        return 1
    else:
        return -1

def first_migration(X,PMc,dim,Xb):
    for j in range(int(PMc)):
        for i in range(dim):
            AI = rng.normal(loc=0, scale=1.2, size=1)
            X[j, i] = X[j, i] + (Xb[i] - X[j, i]) * AI
    return X

def forage(X, Xb, PMc, PMu, dim,):
    for j in range(int(PMc), int(PMc+PMu)):
        for i in range(dim):
            X[j, i] = (X[j, i] + rand_1() * Xb[i] + np.random.randn() * (np.random.randn() * np.abs(Xb[i] + rand_1() * X[j, i]))) / (rng.chisquare(df=8, size=1))
    return X

def second_migration(X, PMc, PMu, pop, dim, Xb):
    for j in range(int(PMc+PMu), pop):
        for i in range(dim):
            A1 = rng.normal(loc=0, scale=1.2, size=1)
            X[j, i] = X[j, i]+(Xb[i]-X[j, i])*A1
    return X

def MSA(pop,dim,lb,ub,Max_iter,fun,MP_b):
    X = initial(pop, dim, lb,ub)                    # Initialize the flamingo population.
    fitness = CaculateFitness(X, fun)               # Calculate fitness values for each flamingo.
    fitness, sortIndex = SortFitness(fitness)       # Sort the fitness values of flamingos.
    X = SortPosition(X, sortIndex)                  # Sort the flamingos.
    GbestScore = fitness[0]                         # The optimal value for the current iteration.
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    for i in range(Max_iter):
        Vs=random.random()
        PMf=int((1-MP_b)*Vs*pop)                     # The number of flamingos migrating in the second stage.
        PMc=MP_b*pop                                 # The number of flamingos that migrate in the first phase.
        Pmu=pop-PMc-PMf                              # The number of flamingos foraging for food.
        Xb = X[0, :]

        X = first_migration(X, PMc, dim, Xb)
        X = forage(X, Xb, PMc, Pmu, dim)
        X = second_migration(X, PMc, Pmu, pop, dim, Xb)
        
        X = BorderCheck(X, lb, ub, pop, dim)                    # Boundary detection.
        fitness = CaculateFitness(X, fun)                       # Calculate fitness values.
        fitness, sortIndex = SortFitness(fitness)               # Sort fitness values.
        X = SortPosition(X, sortIndex)                          # Sort the locations according to fitness.
        if (fitness[0] <= GbestScore):                          # Update the global optimal solution.
            GbestScore = fitness[0]
            GbestPositon[0, :] = X[0, :]
        average_fitness = sum(fitness)/len(fitness)
        best_fitness = fitness[0]
        evolution_data.append({
            "average_fitness": average_fitness[0],
            "best_fitness": best_fitness[0]
        })
    return GbestScore,GbestPositon


'''The main function '''
                            # Set relevant parameters.
time_start = time.time()
pop = 50                    # Flamingo population size.
MaxIter = 300               # Maximum number of iterations.
dim = 20                    # The dimension.
fl=-100                     # The lower bound of the search interval.
ul=100                      # The upper bound of the search interval.
lb = fl*np.ones([dim, 1])
ub = ul*np.ones([dim, 1])
MP_b=0.1                    # The basic proportion of flamingos migration in the first stage.
evolution_data = []
GbestScore, GbestPositon = MSA(pop, dim, lb, ub, MaxIter, fun, MP_b)
time_end = time.time()
print(f"The running time is: {time_end  - time_start } s")
print('The optimal value：',GbestScore)
print('The optimal solution：',GbestPositon)
a_f = [d['average_fitness'] for d in evolution_data]
max_f = [d['best_fitness'] for d in evolution_data]

plt.plot(list(range(len(evolution_data))), a_f)
plt.xlabel("iterations")
plt.ylabel("average value")
plt.title("Average value per iteration")
plt.tight_layout()
plt.fill_between(list(range(len(evolution_data))), a_f)
plt.savefig('average.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

plt.plot(list(range(len(evolution_data))), max_f)
plt.xlabel("iterations")
plt.ylabel("max value")
plt.title("Max value per iteration")
plt.tight_layout()
plt.fill_between(list(range(len(evolution_data))), max_f)
plt.savefig('max.png')
