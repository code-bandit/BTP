# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:48:08 2020

@author: ripud
"""
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, mean_squared_error


X = []
Y = []
x_train = []
y_train = []
x_test = []
y_test = []


class Particle:
    def __init__(self):
        self.position = np.array([(1) ** (bool(random.getrandbits(1))) * random.random()*50, (1) ** (bool(random.getrandbits(1))) * random.random()*50])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0, 0])
        
    def __str__(self):
        print("I am at ", self.position, "my best position is ", self.pbest_position)

    def setParticlePosition(self, position):
        self.position = position
        self.pbest_position = position
    def move(self):
        self.position = self.position + self.velocity
        

"""
speed :

vi(t+1) = w∗vi(t) + c1∗rand1∗(pbesti(t) − xi(t)) + c2∗rand2∗(gbesti(t) − xi(t))
w = 1
c1 = c2 = 2

location:

xi(t+1) = xi(t) + vi(t+1)

"""


class Pso:
    def __init__(self, target, target_error, n_particles, c1 = 2, c2 = 2, w = 1):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*50, random.random()*50])
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
    def printParticles(self):
        for particle in self.particles:
            particle.__str__()
            
    def fitnessFunction(self, particle):
        # return particle.position[0]**2 + particle.position[1]**2 + 1
        clf = OneClassSVM(kernel="rbf", gamma=abs(particle.position[0]))
        clf.fit(x_train)
        y_pred = clf.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    
    def set_pbest(self):
        for particle in self.particles:
            candidate_fitness = self.fitnessFunction(particle)
            if(candidate_fitness < particle.pbest_value):
                particle.pbest_value = candidate_fitness
                particle.pbest_position = particle.position
                
    def set_gbest(self):
        for particle in self.particles:
            best_candidate_fitness = self.fitnessFunction(particle)
            if(self.gbest_value > best_candidate_fitness):
                self.gbest_value = best_candidate_fitness
                self.gbest_position = particle.position
        
    def move_particles(self):
        for particle in self.particles:
            new_velocity = (self.w*particle.velocity) + (self.c1*random.random()) * (particle.pbest_position - particle.position) + (random.random()*self.c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()

#SVM Initialization
def svm(particles_vector, n_particles, x_train, y_train, C, gamma):

    clf = OneClassSVM(kernel="rbf", gamma=gamma)
    clf.fit(x_train)
    # y_pred = clf.predict(x_test)
    # print(accuracy_score(y_test, y_pred))
    return clf




"""
For SVM

"""
def mainPSO(X, Y, target_error=1e-6, n_iterations=30):
    #X
    # for particle in particles_vector:
        # X.append(particle.position)

    #Y
    # for _ in range(n_particles) :
        # Y.append(random.getrandbits(1))

    n_particles = len(X)

    #PSO Initialization
    # target_error = float(input("Enter the target error : "))
    # n_particles = int(input("Enter population size : "))
    # n_iterations = int(input("Enter number of iterations : "))

    pso = Pso(1, target_error, n_particles)
    particles_vector = [Particle() for _ in range(pso.n_particles)]
    pso.particles = particles_vector
    # pso.printParticles()

    i = 0
    for particle in particles_vector:
        particle.setParticlePosition(np.array(X[i]))
        i += 1

    global x_train
    global x_test
    global y_train
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)



    """
    End for SVM

    """

    iteration = 0
    while(iteration < n_iterations):
        pso.set_pbest()
        pso.set_gbest()

        if(abs(pso.gbest_value - pso.target) <= pso.target_error):
            break

        pso.move_particles()
        iteration += 1
        
    print("The best solution is: ", pso.gbest_position, " in n_iterations: ", iteration)


    return svm(particles_vector, n_particles, X, Y, abs(pso.gbest_position[1]), abs(pso.gbest_position[0]))

    """     Testing     """
    #testVar0 = Particle()
    #testVar0.__str__()

if __name__ == "__main__":
    mainPSO(X, Y)