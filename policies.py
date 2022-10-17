import torch
import torch.nn as nn
from abstracts import PyPolicy

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(LinearModel, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
        )

    def forward(self, input):
        main = self.main(input)
        return main

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

class LinearPolicy(PyPolicy):
    def __init__(self, policy_params):
        super(LinearPolicy, self).__init__(policy_params)
        self.model = LinearModel(self.state_dim, self.action_dim)

        # Remove grad from model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def act(self, state):
        state      = self.state_filter(state, update=self.update_filter)
        state      = torch.tensor(state).float()
        prediction = self.model(state)
        action     = prediction.data.numpy()
        
        return action
