import torch
import torch.nn as nn

def get_activation(activation_name):
    """Return the activation function based on the name"""
    activation_functions = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'Tanh': nn.Tanh(),
        'GELU': nn.GELU(),
        'ELU': nn.ELU(),
        'Sigmoid': nn.Sigmoid(),
        'SiLU': nn.SiLU()  # Also known as Swish
    }
    
    if activation_name in activation_functions:
        return activation_functions[activation_name]
    else:
        print(f"Warning: Activation '{activation_name}' not found, using ReLU")
        return nn.ReLU()

def policy_net(input_shape, num_actions, hidden_units_1, hidden_units_2, dropout_rate,
                activation_1, activation_2):
    """Create a customized policy network with given architecture parameters"""
    class PolicyNet(nn.Module):
        def __init__(self):
            super(PolicyNet, self).__init__()
            self.input_shape = input_shape
            self.num_actions = num_actions
            
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_shape[0] * input_shape[1], hidden_units_1),
                get_activation(activation_1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_units_1, hidden_units_2),
                get_activation(activation_2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_units_2, num_actions),
                nn.Softmax(dim=-1)
            )
        
        def forward(self, x):
            probs = self.net(x)
            return probs
    
    return PolicyNet()
"""
class PolicyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNet, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 8),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(16, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        probs = self.net(x)
        return probs
"""
