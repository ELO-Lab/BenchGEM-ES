import numpy as np
from optimizers import SGD
from abstracts import OOOptimizer, BestSolution, GradBuffer

"""=======================================  SIMPLE ES  ======================================="""
class ES(OOOptimizer):
    def __init__(self, 
    		 sigma         = 0.1,
                 learning_rate = 0.01,
                 pop_size      = 256,
                 elite_size    = 256,
                 seed          = 0):

        self.sigma 	   = sigma
        self.learning_rate = learning_rate
        assert(pop_size % 2 == 0), "Population size must be even"
        self.pop_size, self.half_popsize = pop_size, pop_size // 2
        self.elite_size = elite_size if elite_size <= pop_size else pop_size
        self.rg 	= np.random.RandomState(seed)

    def initialize(self, mustart):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()

        self.mu 	= np.copy(mustart)
        self.num_params = len(self.mu)
        self.optimizer  = SGD(self, self.learning_rate)

    def ask(self):
        self.half_epsilon = self.rg.randn(self.half_popsize, self.num_params)
        # Normalize epsilons
        for i in range(self.half_popsize):
            self.half_epsilon[i, :] = (np.sqrt(self.num_params) / np.linalg.norm(self.half_epsilon[i, :])) * self.half_epsilon[i, :]
        self.epsilon   = np.concatenate([self.half_epsilon, -self.half_epsilon])
        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions

    def tell(self, func_vals):
        assert(len(func_vals) == self.pop_size), "Inconsistent population size"
        self.iterations  += 1
        self.evaluations += self.pop_size

        # Save the best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Select elite solutions
        min_vals = np.array([min(func_vals[i], func_vals[i + self.half_popsize]) for i in range(self.half_popsize)])
        idx = np.argsort(min_vals)[: (self.elite_size // 2)]
        idx = np.concatenate([idx, idx + self.half_popsize])
        elite_epsilon   = self.epsilon[idx]
        elite_func_vals = np.array(func_vals)[idx]

        # Normalize values
        elite_func_vals = elite_func_vals / np.std(elite_func_vals)

        # Calculate gradient
        grad = 1./(self.pop_size * self.sigma) * np.dot(elite_epsilon.T, elite_func_vals)

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


"""=======================================  GUIDED ES  ======================================="""
class GES(OOOptimizer):
    def __init__(self, 
    		 sigma         = 0.1,
                 alpha         = 0.5,
                 learning_rate = 0.01,
                 pop_size      = 256,
                 elite_size    = 256,
                 warm_up       = 20,
                 seed	       = 0):

        self.sigma         = sigma
        self.alpha 	   = alpha
        self.learning_rate = learning_rate
        assert(pop_size % 2 == 0), "Population size must be even"
        self.pop_size, self.half_popsize = pop_size, pop_size // 2
        self.elite_size = elite_size if elite_size <= pop_size else pop_size
        self.warm_up    = warm_up
        self.rg 	= np.random.RandomState(seed)

    def initialize(self, mustart, k):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()

        self.mu 	= np.copy(mustart)
        self.num_params = len(self.mu)
        self.optimizer  = SGD(self, self.learning_rate)

        self.k 		= k
        self.surg_grads = GradBuffer(self.k, self.num_params)

    def ask(self):
        if self.iterations <= max(self.k, self.warm_up):
            a = np.sqrt(self.alpha / self.num_params)
            self.half_epsilon = a * self.rg.randn(self.half_popsize, self.num_params)
        else:
            U, _ = np.linalg.qr(self.surg_grads.grads.T)
            a = np.sqrt(self.alpha / self.num_params)
            c = np.sqrt((1 - self.alpha) / self.k)
            half_epsilon_1 = a * self.rg.randn(self.half_popsize, self.num_params)
            half_epsilon_2 = c * self.rg.randn(self.half_popsize, self.k) @ U.T
            self.half_epsilon = half_epsilon_1 + half_epsilon_2
        # Normalize epsilons
        for i in range(self.half_popsize):
            self.half_epsilon[i, :] = (np.sqrt(self.num_params) / np.linalg.norm(self.half_epsilon[i, :])) * self.half_epsilon[i, :]
        self.epsilon   = np.concatenate([self.half_epsilon, -self.half_epsilon])
        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions

    def tell(self, func_vals):
        assert(len(func_vals) == self.pop_size), "Inconsistent population size"
        self.iterations  += 1
        self.evaluations += self.pop_size

        # Save best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Select elite solutions
        min_vals = np.array([min(func_vals[i], func_vals[i + self.half_popsize]) for i in range(self.half_popsize)])
        idx = np.argsort(min_vals)[: (self.elite_size // 2)]
        idx = np.concatenate([idx, idx + self.half_popsize])
        elite_epsilon   = self.epsilon[idx]
        elite_func_vals = np.array(func_vals)[idx]

        # Normalize values
        elite_func_vals = elite_func_vals / np.std(elite_func_vals)

        # Calculate gradient
        grad = 1./(self.pop_size * self.sigma) * np.dot(elite_epsilon.T, elite_func_vals)

        # Update surrogate gradient matrix
        self.surg_grads.add(grad)

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


"""=====================================  SELF-GUIDED ES  ===================================="""
class SGES(OOOptimizer):
    def __init__(self, 
    		 sigma         = 0.1,
                 min_alpha     = 0.1,
                 max_alpha     = 0.8,
                 alpha_step    = 1.05,
                 learning_rate = 0.01,
                 pop_size      = 256,
                 elite_size    = 256,
                 warm_up       = 20,
                 seed	       = 0,
                 adapt_alpha   = True):
    
        self.sigma 	   = sigma
        self.min_alpha     = min_alpha
        self.max_alpha     = max_alpha
        self.alpha_step    = alpha_step
        self.learning_rate = learning_rate
        assert (pop_size % 2 == 0), "Population size must be even"
        self.pop_size, self.half_popsize = pop_size, pop_size // 2
        self.elite_size  = elite_size if elite_size <= pop_size else pop_size
        self.warm_up     = warm_up
        self.rg 	 = np.random.RandomState(seed)
        self.adapt_alpha = adapt_alpha

    def initialize(self, mustart, k):
        self.iterations    = 0
        self.evaluations   = 0
        self.best_solution = BestSolution()

        self.mu 	= np.copy(mustart)
        self.num_params = len(self.mu)
        self.optimizer  = SGD(self, self.learning_rate)

        self.alpha 	= 0.5
        self.k 		= k
        self.surg_grads = GradBuffer(self.k, self.num_params)

    def ask(self):
        if self.iterations <= max(self.k, self.warm_up):
            a = 1. / np.sqrt(self.num_params)
            self.half_epsilon = a * self.rg.randn(self.half_popsize, self.num_params)
        else:
            U, _ = np.linalg.qr(self.surg_grads.grads.T)
            self.choice = self.rg.random(self.half_popsize)
            self.half_epsilon = np.zeros((self.half_popsize, self.num_params))
            for i in range(self.half_popsize):
                if self.choice[i] < self.alpha:
                    a = 1. / np.sqrt(self.k)
                    self.half_epsilon[i, :] = a * self.rg.randn(self.k) @ U.T
                else:
                    a = 1. / np.sqrt(self.num_params)
                    self.half_epsilon[i, :] = a * self.rg.randn(self.num_params)        
        # Normalize epsilons
        for i in range(self.half_popsize):
            self.half_epsilon[i, :] = (np.sqrt(self.num_params) / np.linalg.norm(self.half_epsilon[i, :])) * self.half_epsilon[i, :]
        self.epsilon   = np.concatenate([self.half_epsilon, -self.half_epsilon])
        self.solutions = self.mu.reshape(1, self.num_params) + self.sigma * self.epsilon

        return self.solutions

    def tell(self, func_vals):
        assert(len(func_vals) == self.pop_size), "Inconsistent population size"
        self.iterations  += 1
        self.evaluations += self.pop_size

        # Save best solution
        ibest = func_vals.index(min(func_vals))
        self.best_solution.update(self.solutions[ibest], func_vals[ibest], self.evaluations, self.iterations)

        # Select elite solutions
        min_vals = np.array([min(func_vals[i], func_vals[i + self.half_popsize]) for i in range(self.half_popsize)])
        idx = np.argsort(min_vals)[: (self.elite_size // 2)]
        idx = np.concatenate([idx, idx + self.half_popsize])
        elite_epsilon   = self.epsilon[idx]
        elite_func_vals = np.array(func_vals)[idx]

        # Normalize values
        elite_func_vals = elite_func_vals / np.std(elite_func_vals)

        # Calculate gradient
        grad = 1./(self.pop_size * self.sigma) * np.dot(elite_epsilon.T, elite_func_vals)

        # Update surrogate gradient matrix
        self.surg_grads.add(grad)

        # Update mean
        update_ratio = self.optimizer.update(grad)

        # Adapt alpha by loss of gradient
        if self.adapt_alpha:
            if self.iterations > max(self.k, self.warm_up) + 1:
                grad_loss, random_loss = [], []
                for i in range(self.half_popsize):
                    # Consider choosing min() or max()
                    # We choose min() because of minimization
                    if self.choice[i] < self.alpha:
                        grad_loss.append(min(func_vals[i], func_vals[i + self.half_popsize]))
                    else:
                        random_loss.append(min(func_vals[i], func_vals[i + self.half_popsize]))
                mean_grad_loss   = None if len(grad_loss) == 0 else np.mean(np.asarray(grad_loss))
                mean_random_loss = None if len(random_loss) == 0 else np.mean(np.asarray(random_loss))
            
                if mean_grad_loss and mean_random_loss:
                    self.alpha = self.alpha * self.alpha_step if mean_grad_loss < mean_random_loss else self.alpha / self.alpha_step
                    self.alpha = self.max_alpha if self.alpha > self.max_alpha else self.alpha
                    self.alpha = self.min_alpha if self.alpha < self.min_alpha else self.alpha

    def stop(self, max_iters, max_evals, early_stop=False):
        if early_stop:
            if self.iterations > self.best_solution.get()[3] + int(max_iters / 5):
                return True
        if self.evaluations >= max_evals or self.iterations >= max_iters:
            return True

    def result(self):
        return self.best_solution.get()
