# Adapted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
import numpy as np
from abstracts import Optimizer

class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v        = np.zeros(self.dim, dtype=np.float32)
    self.stepsize = stepsize
    self.momentum = momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step   = -self.stepsize * self.v
    return step

class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1    = beta1
    self.beta2    = beta2
    self.m        = np.zeros(self.dim, dtype=np.float32)
    self.v        = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a      = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step   = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step
