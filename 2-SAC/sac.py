import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils import EMA, Args
from models import GaussianPolicy, QNetwork, ValueNetwork
from replay_memory import ReplayMemory

class SAC:
    def __init__(self, input_dim, action_space, args: Args, memory: ReplayMemory):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_dim = args.hidden_dim
        self.n_network  = args.n_QNetwork
        
        self.memory = memory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space = action_space
        action_dim = action_space.shape[0]

        self.critic = QNetwork(input_dim, action_dim, self.hidden_dim, self.n_network).to(self.device)
        self.policy = GaussianPolicy(input_dim, action_dim, args.hidden_dim, action_space).to(self.device)
        self.value = ValueNetwork(input_dim, args.hidden_dim).to(self.device)
        self.target_value = ValueNetwork(input_dim, args.hidden_dim).to(self.device)

        self.target_value.load_state_dict(self.value.state_dict())

        self.value_optim = Adam(self.value.parameters(), lr=args.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate = False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state, eval=evaluate)
        return action.detach().cpu().numpy()[0]
    
    def update_params(self, batch_size, updates):
        state, action, reward, next_state, done = self.memory.sample_batch(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        mask = torch.FloatTensor(1 - done).to(self.device).unsqueeze(1)

        ## VALUE NETWORK UPDATE
        value_prediction = self.value(state)
        with torch.no_grad():
            new_action, log_prob, _ = self.policy.sample(state, eval=False)
            qs = self.critic(state, new_action)
            min_q = torch.min(torch.stack(qs), dim=0).values
            v_target = min_q - self.alpha * log_prob
        
        V_loss = F.mse_loss(value_prediction, v_target)
        self.value_optim.zero_grad()
        V_loss.backward()
        self.value_optim.step()

        ## Q NETWORK UPDATE
        with torch.no_grad():
            q_target = mask * self.gamma * self.target_value(next_state) + reward
        Q_prediction = self.critic(state, action)

        q_loss = sum(F.mse_loss(q, q_target) for q in Q_prediction)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        ## POLICY UPDATE
        policy_action, log_prob, _ = self.policy.sample(state, eval=False)
        qs_policy = self.critic(state, policy_action)
        min_qs_policy = torch.min(torch.stack(qs_policy), dim=0).values

        policy_loss = (self.alpha * log_prob - min_qs_policy).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ## VALUE NETWORK EMA UPDATE
        EMA(self.target_value, self.value, self.tau)

        return V_loss.item(), q_loss.item(), policy_loss.item()