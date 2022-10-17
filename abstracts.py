"""================================ EVOLUTIONARY STRATEGIES =================================="""
# Adapted from:
# On Object-Oriented Programming of Optimizers - Collette (2010)

class OOOptimizer(object):
    def __init__(self):
        """take an initial point to start with"""
        raise NotImplementedError
    
    def initialize(self):
        raise NotImplementedError

    def ask(self):
        """deliver (one ore more) candidate solution(s) in a list"""
        raise NotImplementedError

    def tell(self, func_vals):
        """update internal state, prepare for next iteration"""
        raise NotImplementedError

    def stop(self):
        """check whether termination is in order, prevent infinite loop"""
        raise NotImplementedError

    def result(self):
        """get best solution, e.g. (x, f, possibly_more)"""
        raise NotImplementedError

    def optimize(self, xstart, objective_function, iterations=2000):
        """find minimizer of objective_function"""
        iteration = 0
        while not self.stop() and iteration < iterations:
            iteration += 1
            X          = self.ask()                             # deliver candidate solutions
            fitvals    = [objective_function(x) for x in X]
            self.tell(X, fitvals)                               # all the work is done here
        return self

class BestSolution(object):
    """keeps track of the best solution"""
    def __init__(self, x=None, f=None, evaluation=0, iterations=0):
        self.x, self.f, self.evaluation, self.iteration = x, f, evaluation, iterations
    def update(self, x, f, evaluations=None, iterations=None):
        if self.f is None or f < self.f:
            self.x, self.f, self.evaluation, self.iteration = x[:], f, evaluations, iterations
    def get(self):
        return self.x, self.f, self.evaluation, self.iteration

"""======================================== OPTIMIZERS ======================================="""
# Adapted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
import numpy as np

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi      = pi
        self.dim     = pi.num_params
        self.epsilon = epsilon
        self.t       = 0

    def update(self, globalg):
        self.t    += 1
        step       = self._compute_step(globalg)
        theta      = self.pi.mu
        ratio      = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError

"""======================================== BENCHMARKS ======================================="""
class ObjectiveFunction(object):
    def __init__(self, dimension, shift=False):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError
    
    def info(self):
        """get fopt and xopt of function"""
        raise NotImplementedError
    
    def _shift(self):
        """shift f and x with some distribution"""
        if self.shift:
            f_shift = np.random.randn()
            x_shift = np.random.randn(self.dimension)
        else:
            f_shift = 0
            x_shift = np.zeros(self.dimension)
            
        return f_shift, x_shift

"""========================================= POLICIES ========================================"""
# Code in this file is copied and adapted from
# https://github.com/modestyachts/ARS/blob/master/code/policies.py
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from filter import get_filter

class PyPolicy(object):
    """ Policy with Pytorch framework.
    """
    def __init__(self, policy_params):
        self.state_dim  = policy_params['state_dim']
        self.action_dim = policy_params['action_dim']

        # A filter for updating statistics of the observations and normalizing inputs to the policies
        self.state_filter  = get_filter(policy_params['filter'], shape=(self.state_dim,))
        self.update_filter = True
        
    def update_params(self, params):
        params = torch.tensor(params).float()
        vector_to_parameters(params, self.model.parameters())

    def get_params(self):
        parameters = parameters_to_vector(self.model.parameters())
        return parameters.numpy()
    
    def save_model(self, output_path):
        torch.save(self.model.state_dict(), output_path)

    def load_model(self, input_path, eval=False):
        self.model.load_state_dict(torch.load(input_path))
        if eval:
            # To set dropout and batch normalization to evaluation mode.
            self.model.eval()

    def get_observation_filter(self):
        return self.state_filter

    def act(self, state):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

"""========================================= BUFFERS ========================================="""
class GradBuffer(object):
    def __init__(self, max_size, grad_dim):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.grads = np.zeros((max_size, grad_dim))

    def add(self, grad):
        self.grads[self.ptr] = grad
        self.ptr             = (self.ptr + 1) % self.max_size
        self.size            = min(self.size + 1, self.max_size)
