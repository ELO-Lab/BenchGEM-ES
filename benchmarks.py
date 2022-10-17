# Adopted from: 
# https://github.com/CMA-ES/pycma/blob/master/cma/fitness_functions.py
import numpy as np
from abstracts import ObjectiveFunction

class Sphere(ObjectiveFunction):
    def __init__(self, dimension, shift=False):
        self.dimension = dimension
        self.shift     = shift
        self.f_shift, self.x_shift = self._shift()
    
    def __call__(self, x):
        x = np.copy(x) - self.x_shift
        return sum(np.asarray(x)**2) + self.f_shift
    
    def info(self):
        return (self.f_shift, self.x_shift, "Sphere")

class Rosenbrock(ObjectiveFunction):
    def __init__(self, dimension, shift=False):
        self.dimension = dimension
        self.shift     = shift
        self.f_shift, self.x_shift = self._shift()
        self.alpha = 1e2
    
    def __call__(self, x):
        x = np.copy(x) - self.x_shift
        x = [x] if np.isscalar(x[0]) else x
        x = np.asarray(x)
        f = [sum(self.alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
        return (f + self.f_shift) if len(f) > 1 else (f[0] + self.f_shift)
    
    def info(self):
        return (self.f_shift, self.x_shift + np.ones(self.dimension), "Rosenbrock")

class Rastrigin(ObjectiveFunction):
    def __init__(self, dimension, shift=False):
        self.dimension = dimension
        self.shift     = shift
        self.f_shift, self.x_shift = self._shift()
    
    def __call__(self, x):
        x = np.copy(x) - self.x_shift
        if not np.isscalar(x[0]):
            N = len(x[0])
            return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
        N = len(x)
        return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)) + self.f_shift
    
    def info(self):
        return (self.f_shift, self.x_shift, "Rastrigin")

class Ackley(ObjectiveFunction):
    def __init__(self, dimension, shift=False):
        self.dimension = dimension
        self.shift     = shift
        self.f_shift, self.x_shift = self._shift()
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
    
    def __call__(self, x):
        x = np.copy(x) - self.x_shift
        return -self.a * np.exp(-self.b * np.sqrt(1./self.dimension * np.sum(x * x))) \
                - np.exp(1./self.dimension * np.sum(np.cos(self.c * x))) + self.a + np.exp(1)
    
    def info(self):
        return (self.f_shift, self.x_shift, "Ackley")


class Manager(object):
    def __init__(self, dimension, seed=2022, shift=False):
        self.dimension = dimension
        self.shift     = shift
        self.rng       = np.random.RandomState(seed)
    
    def get_initial_solution(self):
        return self.rng.randn(self.dimension)
    
    def get_function(self, id):
        if id == 1:
            return Sphere(self.dimension, self.shift)
        elif id == 2:
            return Rosenbrock(self.dimension, self.shift)
        elif id == 3:
            return Rastrigin(self.dimension, self.shift)
        elif id == 4:
            return Lunacek(self.dimension, self.shift)        
        elif id == 5:
            return Ackley(self.dimension, self.shift)
