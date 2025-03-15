import torch
import torch.nn as nn

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
