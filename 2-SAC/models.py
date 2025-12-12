import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(_init_weight)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, num_network = 2):
        super(QNetwork, self).__init__()

        self.networks = nn.ModuleList()
        for _ in range(num_network):
            network = nn.Sequential(
                nn.Linear(input_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            network.apply(_init_weight)
            self.networks.append(network)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        q = [network(x) for network in self.networks]

        return tuple(q)
    

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, action_space = None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std   = nn.Linear(hidden_dim, action_dim)

        self.apply(_init_weight)

        # `register_buffer`: Non-learnable parameter but moves to the device automatically.
        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.))
            self.register_buffer('action_bias', torch.tensor(0.))
        else:
            high = torch.FloatTensor(action_space.high)
            low = torch.FloatTensor(action_space.low)
            self.register_buffer('action_scale', (high - low) / 2.0)
            self.register_buffer('action_bias', (high + low) / 2.0)

    def forward(self, state, eval = False):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        if eval:
            return mean, None
            
        log_std = torch.clamp(self.log_std(x), min = LOG_SIG_MIN, max = LOG_SIG_MAX)
        return mean, log_std
        
    def sample(self, state, eval = False):
        mean, log_std = self.forward(state, eval = eval)
        if not eval:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)

            action = y_t * self.action_scale + self.action_bias

            # P(y) = P(x) * |dx / dy|
            # log P(y) = log P(x) - log |dy/dx| = log P(x) - (1 - y^2) * action_scale
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim = True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean
        else:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, torch.tensor(0.), action
    