import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import wandb
import time

# --- 1. Define the Models (Your v3-tanh-fix version) ---

class ActorModel(nn.Module):
    """
    Actor Network (The Policy)
    Takes a state and decides on a mean action.
    """
    def __init__(self, lr, input_size, output_size, max_action):
        super(ActorModel, self).__init__()
        self.lr = lr
        self.h = 256
        self.max_action = max_action
        
        self.linear1 = nn.Linear(input_size, self.h)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.h, self.h)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.h, output_size)
        self.tanh = nn.Tanh()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fnc = nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        # Squash output to [-1, 1] and scale by max_action
        x = self.tanh(x)
        x = x * self.max_action
        return x

    def update_weights(self, prediction, target):
        loss = self.loss_fnc(prediction, target.detach()) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CriticModel(nn.Module):
    """
    Critic Network (The Value Function)
    Takes a state and estimates the expected future reward (the V-value).
    """
    def __init__(self, lr, input_size):
        super(CriticModel, self).__init__()
        self.lr = lr
        self.h = 256
        
        self.linear1 = nn.Linear(input_size, self.h)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.h, self.h)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.h, 1) # Outputs a single value

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 
        self.loss_fnc = nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

    def update_weights(self, prediction, target):
        loss = self.loss_fnc(prediction, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- 2. Main Training Function ---
def main():
    # Initialize WandB for logging results
    wandb.init(project="cacla-vs-cleanrl-benchmark", name="cacla-run")
    
    # Initialize Environment
    env = gym.make("Pendulum-v1")
    
    # Get environment parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Hyperparameters
    actor_lr = 1e-4
    critic_lr = 3e-4
    gamma = 0.99
    sigma_start = 1.0
    sigma_end = 0.1
    sigma_decay = 500 # Faster decay
    total_episodes = 2000 # Run for a fixed number of episodes
    
    # Initialize Actor and Critic
    actor = ActorModel(lr=actor_lr, input_size=state_dim, output_size=action_dim, max_action=max_action)
    critic = CriticModel(lr=critic_lr, input_size=state_dim)
    
    wandb.watch(actor, log="all")
    wandb.watch(critic, log="all")

    print(f"Starting CACLA training on Pendulum-v1...")
    
    for episode in range(total_episodes): 
        state, info = env.reset()
        sigma = sigma_end + (sigma_start - sigma_end) * np.exp(-1. * episode / sigma_decay)
        
        total_reward = 0
        terminated = False
        truncated = False
        t = 0 # Timestep counter
        
        while not terminated and not truncated:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # 1. Get Action
            action_mean = actor(state_tensor)
            action_explored = torch.distributions.Normal(action_mean, sigma).sample()
            action_clipped = action_explored.clamp(-max_action, max_action)
            
            # 2. Take Action
            action_for_env = action_clipped.detach().numpy().flatten()
            next_state, reward, terminated, truncated, info = env.step(action_for_env)
            
            total_reward += reward
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # 3. Update Critic
            with torch.no_grad():
                value_new = critic(next_state_tensor)
            value = critic(state_tensor)
            
            if terminated:
                target = torch.tensor([[reward]], dtype=torch.float32) 
            else:
                target = reward + gamma * value_new
            
            critic.update_weights(prediction=value, target=target)
            
            # 4. Update Actor (CACLA)
            td_error = target - value
            if td_error.item() > 0.0:
                actor.update_weights(target=action_explored, prediction=action_mean)

            state = next_state
            t += 1

        # Log results for this episode
        if episode % 10 == 0:
            print(f"Episode: {episode}, Steps: {t}, Sigma: {sigma:.3f}, Reward: {total_reward:.2f}")
        
        wandb.log({
            "episode": episode, 
            "reward": total_reward, # Use "reward" for easy comparison
            "steps": t,
            "sigma": sigma
        })

    env.close()

if __name__ == "__main__":
    main()