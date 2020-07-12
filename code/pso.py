# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:48:08 2020

@author: ripud
"""
import random
import numpy as np


class Particle:
    def __init__(self):
        self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1) ** (bool(random.getrandbits(1))) * random.random()*50])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0, 0])
        
    def __str__(self):
        print("I am at ", self.position, "my best position is ", self.pbest_position)
        
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
        return particle.position[0]**2 + particle.position[1]**2 + 1
    
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

target_error = float(input("Enter the target error : "))
n_particles = int(input("Enter population size : "))
n_iterations = int(input("Enter number of iterations : "))

search_space = Pso(1, target_error, n_particles)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.printParticles()

iteration = 0
while(iteration < n_iterations):
    search_space.set_pbest()
    search_space.set_gbest()

    if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break

    search_space.move_particles()
    iteration += 1
    
print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)


"""     Testing     """
#testVar0 = Particle()
#testVar0.__str__()
