import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from TD3_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 4e-2              # for soft update of target parameters
LR_ACTOR = 6e-4         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUM_LEARN = 8          # number of learning 
NUM_TIME_STEP = 10      # every NUM_TIME_STEP do update
EPSILON = 1.0           # epsilon to noise of action
EPSILON_DECAY = 2e-5    # epsilon decay to noise epsilon of action
POLICY_DELAY = 3        # delay for policy update (TD3)
AGENT_NUM = 2           # number of agent 

class MultiAgent():    
    def __init__(self, env, state_size, action_size, random_seed=15, device=device):

        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.policy_delay = POLICY_DELAY

        self.agents = []
        self.agent_num = AGENT_NUM
        for _ in range(self.agent_num): self.agents.append(
            Agent(self.state_size, self.action_size, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, EPSILON, EPSILON_DECAY, self.device, self.agent_num)
        )

        self.iterations = np.zeros((self.agent_num,))

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def transform_to_full(self, state, action, reward, next_state, done):
        # transform to full
        # state = (agent_num, state_size) -> full_state = (agent_num*state_size,)
        # action = (agent_num, action_size) -> full_action = (agent_num*action_size,)
        # next_state = (agent_num, state_size) -> full_next_state = (agent_num*state_size,)
        # reward = (agent_num, 1) -> full_reward = (agent_num*1,)
        # done = (agent_num, 1) -> full_done = (agent_num*1,)

        return np.reshape(state, (-1,)), np.reshape(action, (-1,)), np.reshape(reward, (-1,)), np.reshape(next_state, (-1,)), np.reshape(done, (-1,))
                
    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        # transform to full
        full_state, full_action, full_reward, full_next_state, full_done = self.transform_to_full(state, action, reward, next_state, done)
        self.memory.add(full_state, full_action, full_reward, full_next_state, full_done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t % NUM_TIME_STEP == 0:
            for _ in range(NUM_LEARN):
                for agent_idx in range(self.agent_num):
                    experiences = self.memory.sample()
                    self.iterations[agent_idx] += 1
                    self.learn(experiences, agent_idx, GAMMA)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def learn(self, experiences, agent_idx, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # prepare local and full 
        full_states, full_actions, full_rewards, full_next_states, full_dones = experiences
        local_rewards = full_rewards[:, agent_idx:(agent_idx+1)]
        local_dones = full_dones[:, agent_idx:(agent_idx+1)]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            actions_next = torch.cat([self.agents[idx].actor_target(full_next_states[:, idx*self.state_size:(idx+1)*self.state_size]) for idx in range(self.agent_num)], dim=-1) # (batch_size, action_size*agent_num)
            actions_next += torch.clip(torch.normal(mean=0., std=0.2, size=actions_next.shape), -1.0, 1.0).to(self.device) ##################### TD3
            actions_next = torch.clip(actions_next, -1.0, 1.0)
            Q_targets_next = torch.min(self.agents[agent_idx].critic_target_1(full_next_states, actions_next), self.agents[agent_idx].critic_target_2(full_next_states, actions_next)) ##################### TD3

        # Compute Q targets for current states (y_i)
        Q_targets = local_rewards + (gamma * Q_targets_next * (1 - local_dones))

        # Compute critic loss
        Q_expected_1 = self.agents[agent_idx].critic_local_1(full_states, full_actions)
        Q_expected_2 = self.agents[agent_idx].critic_local_2(full_states, full_actions)
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets.detach())
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets.detach())

        # Minimize the loss
        self.agents[agent_idx].critic_optimizer_1.zero_grad()
        self.agents[agent_idx].critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic_local_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic_local_2.parameters(), 1.0)
        self.agents[agent_idx].critic_optimizer_1.step()
        self.agents[agent_idx].critic_optimizer_2.step()

        if self.iterations[agent_idx] % self.policy_delay == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = torch.cat([self.agents[idx].actor_local(full_states[:, idx*self.state_size:(idx+1)*self.state_size]) if idx == agent_idx
             else self.agents[idx].actor_local(full_states[:, idx*self.state_size:(idx+1)*self.state_size]).detach() for idx in range(self.agent_num)], dim=-1) # (batch_size, action_size*agent_num)
            actor_loss = -self.agents[agent_idx].critic_local_1(full_states, actions_pred).mean()
            # Minimize the loss
            self.agents[agent_idx].actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].actor_local.parameters(), 1.0)
            self.agents[agent_idx].actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.agents[agent_idx].critic_local_1, self.agents[agent_idx].critic_target_1, TAU)
            self.soft_update(self.agents[agent_idx].critic_local_2, self.agents[agent_idx].critic_target_2, TAU)
            self.soft_update(self.agents[agent_idx].actor_local, self.agents[agent_idx].actor_target, TAU)                 
        
        self.agents[agent_idx].epsilon -= self.agents[agent_idx].epsilon_decay    

    def act(self, states, add_noise=True):
        actions = np.concatenate([agent.act(np.expand_dims(state, axis=0), add_noise) for agent, state in zip(self.agents, states)], axis=0)
        return actions
    
    def noise_reset(self):
        for idx in range(self.agent_num): self.agents[idx].noise.reset()

    def train(self, n_episodes=1800, max_t=3000):
        """
        DDPG.    
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
        """
        self.scores = []                                                # list containing scores from each episode
        max_scores_window = deque(maxlen=100)                               # last 100 scores
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name] # reset the environment
            state = env_info.vector_observations[:, -8:]                        # get the current state
            score = np.zeros(self.agent_num)
            self.noise_reset()                                         # reset OU noise
            for t in range(max_t):
                action = self.act(state, add_noise=True)
                env_info = self.env.step(action)[self.brain_name]       # send the action to the environment
                next_state, reward, done = env_info.vector_observations[:, -8:], env_info.rewards, env_info.local_done
                self.step(state, action, reward, next_state, done, t)
                state = next_state
                score += reward
                if any(done):
                    break 
                        
            self.iterations = np.zeros((self.agent_num,))               # reset iteration
            max_scores_window.append(np.max(score))                     # save most recent score
            self.scores.append(np.max(score))                           # save most recent score
            print('\rEpisode {}\tAverage Score : {:.4f} \t eps : {:.3f}'.format(i_episode, np.mean(max_scores_window), self.agents[0].epsilon), end="")
            if i_episode % 50 == 0:
                print('\rEpisode {}\tAverage Score : {:.4f} \t eps : {:.3f}'.format(i_episode, np.mean(max_scores_window), self.agents[0].epsilon))
            if np.mean(max_scores_window) >= 0.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.34}'.format(i_episode-100, np.mean(max_scores_window)))
                torch.save(self.agents[0].actor_local.state_dict(), '../saved_model/saved_agent_1_TD3_actor.pth')
                torch.save(self.agents[0].critic_local_1.state_dict(), '../saved_model/saved_agent_1_TD3_critic_1.pth')
                torch.save(self.agents[0].critic_local_2.state_dict(), '../saved_model/saved_agent_1_TD3_critic_2.pth')
                torch.save(self.agents[1].actor_local.state_dict(), '../saved_model/saved_agent_2_TD3_actor.pth')
                torch.save(self.agents[1].critic_local_1.state_dict(), '../saved_model/saved_agent_2_TD3_critic_1.pth')
                torch.save(self.agents[1].critic_local_2.state_dict(), '../saved_model/saved_agent_2_TD3_critic_2.pth')
                break
        return self.scores


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

#%%
import torch

a = torch.zeros((64, 24))
b = torch.cat([a, a], dim=-1)
b.shape
