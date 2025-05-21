import wandb
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.data import WeightedRandomSampler  # for stratified sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# Actor Network: Generates continuous actions (normalized in [-1, 1])
# ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
    def forward(self, state):
        return self.net(state)

# ------------------------
# Critic Network: Takes state and action as input and outputs a Q-value (scalar)
# ------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# ------------------------
# Transition Model: Given a state and an action, predicts the next state and reward
# ------------------------
class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(TransitionModel, self).__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim + 1)  # Output: predicted next state and reward
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        out = self.net(x)
        next_state_pred = out[:, :state.shape[1]]
        reward_pred = out[:, state.shape[1]:]
        return next_state_pred, reward_pred

# ------------------------
# Soft Update: Move target network parameters closer to the main network parameters
# ------------------------
def soft_update(net, target_net, tau=0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

# ------------------------
# MBMFAlgorithm Class: Encapsulates the MBMF algorithm (DDPG style + model-based data augmentation)
# ------------------------
class MBMFAlgorithm:
    def __init__(self, env, gamma=0.99, lr=1e-3, actor_lr=1e-4, tau=0.005,
                 buffer_size=500000, batch_size=64, model_batch_size=64,
                 model_planning_steps=3, noise_scale=0.1,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000,
                 l2_reg=1e-4):  # Added L2 regularization parameter
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.model_planning_steps = model_planning_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]  # Continuous actions
        
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.model_network = TransitionModel(self.state_dim, self.action_dim).to(device)
        
        # Use weight_decay parameter to implement L2 regularization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=l2_reg)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=l2_reg)
        self.model_optimizer = optim.Adam(self.model_network.parameters(), lr=lr, weight_decay=l2_reg)
        
        # For tracking the best model
        self.best_reward = -float('inf')
    
    def _get_epsilon(self, step):
        return max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (step / self.epsilon_decay))
    
    def select_action(self, state, noise=True):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            action += self.noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)
    
    def train(self, num_episodes, max_steps_per_episode=200, wandb=wandb, best_model_path="best_model.pth"):
        for episode in range(num_episodes):
            obs_reset = self.env.reset()
            state = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            for t in range(max_steps_per_episode):
                self.total_steps += 1
                epsilon = self._get_epsilon(self.total_steps)
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state, noise=True)
                next_step = self.env.step(action)
                if len(next_step) == 5:
                    next_state, reward, terminated, truncated, _ = next_step
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = next_step
                next_state = np.array(next_state, dtype=np.float32)
                # Store transition with episode number
                self.replay_buffer.append((episode, state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                if len(self.replay_buffer) >= self.batch_size:
                    self._update()
                if done:
                    break
                wandb.log({
                    "epoch": episode+1,
                    "reward": reward,
                    "total_reward": episode_reward,
                    "id_error": next_state[0] - next_state[2],
                    "iq_error": next_state[1] - next_state[3],
                    "vd": action[0],
                    "vq": action[1],
                })
            print(f"Episode {episode}: Total Reward = {episode_reward}")
            # Save model if the current episode's total reward is higher
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save(best_model_path)
                print(f"Best model updated at episode {episode} with reward {episode_reward}")
    
    def _update(self):
        # --- Use WeightedRandomSampler for stratified sampling ---
        transitions = list(self.replay_buffer)  # Each element: (episode, state, action, reward, next_state, done)
        # Count the number of samples per episode
        episode_counts = {}
        for trans in transitions:
            ep = trans[0]
            episode_counts[ep] = episode_counts.get(ep, 0) + 1
        # Calculate the weight for each sample (1 / number of samples in that episode)
        weights = [1.0 / episode_counts[trans[0]] for trans in transitions]
        sampler = WeightedRandomSampler(weights, num_samples=self.batch_size, replacement=True)
        indices = list(sampler)
        batch = [transitions[i] for i in indices]
        # Unpack batch (ignoring the episode index)
        _, states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)
        
        # Update Critic network (using real data)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        soft_update(self.critic, self.critic_target, self.tau)
        
        # Update Actor network (policy gradient)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.actor, self.actor_target, self.tau)
        
        # Update Transition Model (using real data)
        # Here we still use random sampling; you can also replace it with WeightedRandomSampler if needed
        batch_model = random.sample(list(self.replay_buffer), self.batch_size)
        _, s_model, a_model, r_model, s_next_model, _ = zip(*batch_model)
        s_model = torch.tensor(np.array(s_model), dtype=torch.float32, device=device)
        a_model = torch.tensor(np.array(a_model), dtype=torch.float32, device=device)
        r_model = torch.tensor(np.array(r_model), dtype=torch.float32, device=device).unsqueeze(1)
        s_next_model = torch.tensor(np.array(s_next_model), dtype=torch.float32, device=device)
        pred_s_next, pred_r = self.model_network(s_model, a_model)
        model_loss = nn.MSELoss()(pred_s_next, s_next_model) + nn.MSELoss()(pred_r, r_model)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        
        # Use the model to generate virtual data to update the Critic network (using original random sampling logic)
        virtual_batch = []
        for _ in range(self.model_planning_steps):
            batch_virtual = random.sample(list(self.replay_buffer), self.model_batch_size)
            _, s_virtual, _, _, _, _ = zip(*batch_virtual)
            s_virtual = torch.tensor(np.array(s_virtual), dtype=torch.float32, device=device)
            with torch.no_grad():
                a_virtual = self.actor_target(s_virtual)
                s_next_pred, r_pred = self.model_network(s_virtual, a_virtual)
            # Treat virtual transitions as non-terminal
            for i in range(self.model_batch_size):
                virtual_batch.append((s_virtual[i].cpu().numpy(),
                                      a_virtual[i].cpu().numpy(),
                                      r_pred[i].item(),
                                      s_next_pred[i].cpu().numpy(),
                                      0))
        if len(virtual_batch) >= self.batch_size:
            batch_virtual = random.sample(virtual_batch, self.batch_size)
            s_v, a_v, r_v, s_next_v, d_v = zip(*batch_virtual)
            s_v = torch.tensor(np.array(s_v), dtype=torch.float32, device=device)
            a_v = torch.tensor(np.array(a_v), dtype=torch.float32, device=device)
            r_v = torch.tensor(np.array(r_v), dtype=torch.float32, device=device).unsqueeze(1)
            s_next_v = torch.tensor(np.array(s_next_v), dtype=torch.float32, device=device)
            d_v = torch.tensor(np.array(d_v), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                next_a_v = self.actor_target(s_next_v)
                target_q_v = self.critic_target(s_next_v, next_a_v)
                y_v = r_v + self.gamma * (1 - d_v) * target_q_v
            current_q_v = self.critic(s_v, a_v)
            critic_loss_v = nn.MSELoss()(current_q_v, y_v)
            self.critic_optimizer.zero_grad()
            critic_loss_v.backward()
            self.critic_optimizer.step()
            soft_update(self.critic, self.critic_target, self.tau)
    
    def predict(self, obs, deterministic=True):
        obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        return np.clip(action, -1, 1)
    
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "model_network": self.model_network.state_dict()
        }, path)
    
    def load(self, path):
        data = torch.load(path, map_location=device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.model_network.load_state_dict(data["model_network"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
