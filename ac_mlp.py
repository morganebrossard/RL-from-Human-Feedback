import torch.nn as nn
from torch.distributions import Categorical

class ActorMLP(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorMLP, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)  # Probability distribution for equities
        )

    def forward(self, x):
        x = self.fc(x)
        dist = Categorical(x)
        return dist


class CriticMLP(nn.Module):
    def __init__(self, input_dim):
        super(CriticMLP, self).__init__()
        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Scalar output for state value estimation
        )

    def forward(self, x):
        x = self.fc(x)
        return x
