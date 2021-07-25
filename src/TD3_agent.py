import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, lr_actor, lr_critic, weight_decay, epsilon, epsilon_decay, device, agent_num, random_seed=15):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.device = device
        self.agent_num = agent_num

        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.device = device
        self.seed = random.seed(random_seed)


        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local_1 = Critic(self.state_size, self.action_size, random_seed, self.agent_num, fcs1_units=256, fc2_units=128).to(device)
        self.critic_target_1 = Critic(self.state_size, self.action_size, random_seed, self.agent_num, fcs1_units=256, fc2_units=128).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        self.critic_local_2 = Critic(self.state_size, self.action_size, random_seed, self.agent_num, fcs1_units=256, fc2_units=128).to(device)
        self.critic_target_2 = Critic(self.state_size, self.action_size, random_seed, self.agent_num, fcs1_units=256, fc2_units=128).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(self.action_size, random_seed)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
    
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state * 0.5