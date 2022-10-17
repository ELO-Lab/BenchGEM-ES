import numpy as np
import multiprocessing.dummy as mp
from optimizers import BasicSGD
from abstracts import OOOptimizer, BestSolution

"""=======================================  SIMPLE ES  ======================================="""
class ES(OOOptimizer):
    def __init__(self,
                sigma         = 0.1,
                learning_rate = 0.01,
                popsize	      = 256,
                antithetic    = False):

        self.sigma 	   = sigma
        self.learning_rate = learning_rate
        self.popsize 	   = popsize
        self.antithetic    = antithetic
        if self.antithetic:
            assert(self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)
    
    def initialize(self, xstart, obj_func):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()
        self.trajectory    = [np.copy(xstart)]
        self.history_loss  = [obj_func(np.copy(xstart))]

        self.mu 	= np.copy(xstart)
        self.num_params = len(self.mu)
        self.optimizer  = BasicSGD(self, self.learning_rate)
    
    def ask(self):
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon      = np.concatenate([self.epsilon_half, -self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)
        self.epsilon /= np.sqrt(self.num_params)
        
        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions
    
    def tell(self, func_vals):
        assert(len(func_vals) == self.popsize), "Inconsistent population size"
        self.iterations  += 1
        self.evaluations += len(func_vals)

        # Save best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Calculate gradient
        grad = 1./(self.popsize * self.sigma) * np.dot(self.epsilon.T, func_vals)

        # Update mean
        update_ratio = self.optimizer.update(grad)
    
    def stop(self, max_iters, max_evals, early_stop=False):
        if early_stop:
            if self.iterations > self.best_solution.get()[3] + int(max_iters / 5):
                return True
        if self.evaluations >= max_evals or self.iterations >= max_iters:
            return True
    
    def result(self):
        return self.best_solution.get()
    
    def optimize(self, xstart, obj_func, max_iters=10, max_evals=10, early_stop=False, threads=1):
        pool = mp.Pool(threads)
        self.initialize(xstart, obj_func)
        while not self.stop(max_iters, max_evals, early_stop):
            X = self.ask()
            func_vals = pool.map(obj_func, X)
            self.tell(func_vals)
            self.trajectory.append(np.copy(self.mu))
            self.history_loss.append(obj_func(np.copy(self.mu)))
            if self.iterations % 100 == 0:
                print("Iters: %d, Loss: %f" % (self.iterations, self.history_loss[-1]))


"""=======================================  GUIDED ES  ======================================="""
class GES(OOOptimizer):
    def __init__(self,
                sigma         = 0.1,
                alpha	      = 0.5,
                learning_rate = 0.01,
                popsize       = 256):
        
        self.sigma 	   = sigma
        self.alpha 	   = alpha
        self.learning_rate = learning_rate
        assert(popsize % 2 == 0), "Population size must be even"
        self.popsize = popsize

    def initialize(self, xstart, k, obj_func):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()
        self.trajectory    = [np.copy(xstart)]
        self.history_loss  = [obj_func(np.copy(xstart))]

        self.mu 	= np.copy(xstart)
        self.num_params = len(self.mu)
        self.optimizer  = BasicSGD(self, self.learning_rate)

        self.k 		= k
        self.surg_grads = []
    
    def ask(self):
        if self.iterations <= self.k:
            a = np.sqrt(1. / self.num_params)
            self.epsilon_half = a * np.random.randn(self.popsize // 2, self.num_params)
        else:
            U, _ = np.linalg.qr(np.array(self.surg_grads).T)
            a    = np.sqrt(self.alpha / self.num_params)
            c    = np.sqrt((1 - self.alpha) / self.k)
            epsilon_half_1 = a * np.random.randn(self.popsize // 2, self.num_params)
            epsilon_half_2 = c * np.random.randn(self.popsize // 2, self.k) @ U.T
            self.epsilon_half = epsilon_half_1 + epsilon_half_2
        self.epsilon = np.concatenate([self.epsilon_half, -self.epsilon_half])

        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions
    
    def tell(self, func_vals):
        assert(len(func_vals) == self.popsize), "Inconsistent population size"
        self.iterations  += 1
        self.evaluations += len(func_vals)

        # Save best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Calculate gradient
        func_vals = np.array(func_vals)
        grad      = 1./(self.popsize * self.sigma) * np.dot(self.epsilon.T, func_vals)

        # Update mean
        update_ratio = self.optimizer.update(grad)

        # Update surrogate gradient matrix
        if self.iterations <= self.k:
            self.surg_grads.append(grad)
        else:
            self.surg_grads.pop(0)
            self.surg_grads.append(grad)
        
    def stop(self, max_iters, max_evals, early_stop=False):
        if early_stop:
            if self.iterations > self.best_solution.get()[3] + int(max_iters / 5):
                return True
        if self.evaluations >= max_evals or self.iterations >= max_iters:
            return True
    
    def result(self):
        return self.best_solution.get()

    def optimize(self, xstart, k, obj_func, max_iters=10, max_evals=10, early_stop=False, threads=1):
        pool = mp.Pool(threads)
        self.initialize(xstart, k, obj_func)
        while not self.stop(max_iters, max_evals, early_stop):
            X = self.ask()
            func_vals = pool.map(obj_func, X)
            self.tell(func_vals)
            self.trajectory.append(np.copy(self.mu))
            self.history_loss.append(obj_func(np.copy(self.mu)))
            if self.iterations % 100 == 0:
                print("Iters: %d, Loss: %f" % (self.iterations, self.history_loss[-1]))


"""=====================================  SELF-GUIDED ES  ===================================="""
class SGES(OOOptimizer):
    def __init__(self,
                sigma         = 0.1,
                min_alpha     = 0.3,
                max_alpha     = 0.7,
                alpha_step    = 1.005,
                learning_rate = 0.01,
                popsize	      = 256,
                adapt_alpha   = True):
    
        self.sigma         = sigma
        self.min_alpha     = min_alpha
        self.max_alpha     = max_alpha
        self.alpha_step    = alpha_step
        self.learning_rate = learning_rate
        assert (popsize % 2 == 0), "Population size must be even"
        self.popsize = popsize
        self.adapt_alpha = adapt_alpha

    def initialize(self, xstart, k, obj_func):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()
        self.trajectory    = [np.copy(xstart)]
        self.history_loss  = [obj_func(np.copy(xstart))]

        self.mu         = np.copy(xstart)
        self.num_params = len(self.mu)
        self.optimizer  = BasicSGD(self, self.learning_rate)

        self.alpha      = 0.5
        self.k          = k
        self.surg_grads = []
    
    def ask(self):
        if self.iterations <= self.k:
            a = 1. / np.sqrt(self.num_params)
            self.epsilon_half = a * np.random.randn(self.popsize // 2, self.num_params)
        else:
            U, _ 	      = np.linalg.qr(np.array(self.surg_grads).T)
            self.choice       = np.random.random(self.popsize // 2)
            self.epsilon_half = np.zeros((self.popsize // 2, self.num_params))
            for i in range(self.popsize // 2):
                if self.choice[i] < self.alpha:
                    a = 1. / np.sqrt(self.k)
                    self.epsilon_half[i, :] = a * np.random.randn(self.k) @ U.T
                else:
                    a = 1. / np.sqrt(self.num_params)
                    self.epsilon_half[i, :] = a * np.random.randn(self.num_params)
        self.epsilon = np.concatenate([self.epsilon_half, -self.epsilon_half])

        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions

    def tell(self, func_vals):
        self.iterations  += 1
        self.evaluations += len(func_vals)

        # Save best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Calculate gradient
        func_vals = np.array(func_vals)
        grad      = 1./(self.popsize * self.sigma) * np.dot(self.epsilon.T, func_vals)

        # Update mean
        update_ratio = self.optimizer.update(grad)

        # Adapt alpha by loss of gradient
        if self.iterations > self.k + 1:
            grad_loss, random_loss = [], []
            for i in range(self.popsize // 2):
                # Because of minimizing, we will choose max() instead of min()
                if self.choice[i] < self.alpha:
                    grad_loss.append(min(func_vals[i], func_vals[i + self.popsize // 2]))
                else:
                    random_loss.append(min(func_vals[i], func_vals[i + self.popsize // 2]))      
            mean_grad_loss   = 10000 if grad_loss is None else np.mean(np.asarray(grad_loss))
            mean_random_loss = 10000 if random_loss is None else np.mean(np.asarray(random_loss))

            if self.adapt_alpha:
                self.alpha = self.alpha * self.alpha_step if mean_grad_loss < mean_random_loss else self.alpha / self.alpha_step
                self.alpha = self.max_alpha if self.alpha > self.max_alpha else self.alpha
                self.alpha = self.min_alpha if self.alpha < self.min_alpha else self.alpha

        # Update surrogate gradient matrix
        if self.iterations <= self.k:
            self.surg_grads.append(grad)
        else:
            self.surg_grads.pop(0)
            self.surg_grads.append(grad)

    
    def stop(self, max_iters, max_evals, early_stop=False):
        if early_stop:
            if self.iterations > self.best_solution.get()[3] + int(max_iters / 5):
                return True
        if self.evaluations >= max_evals or self.iterations >= max_iters:
            return True
    
    def result(self):
        return self.best_solution.get()

    def optimize(self, xstart, k, obj_func, max_iters=10, max_evals=10, early_stop=False, threads=1):
        pool = mp.Pool(threads)
        self.initialize(xstart, k, obj_func)
        while not self.stop(max_iters, max_evals, early_stop):
            X = self.ask()
            func_vals = pool.map(obj_func, X)
            self.tell(func_vals)
            self.trajectory.append(np.copy(self.mu))
            self.history_loss.append(obj_func(np.copy(self.mu)))
            if self.iterations % 100 == 0:
                print("Iters: %d, Loss: %f" % (self.iterations, self.history_loss[-1]))
