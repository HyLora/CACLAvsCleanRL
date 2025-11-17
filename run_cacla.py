import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import wandb
import time
# --- ADDED: Import the video recorder wrapper ---
import gym.wrappers

# --- 1. Define the Models (Same as before) ---

class ActorModel(nn.Module):
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
        x = self.tanh(x)
        x = x * self.max_action
        return x

    def update_weights(self, prediction, target):
        loss = self.loss_fnc(prediction, target.detach()) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CriticModel(nn.Module):
    def __init__(self, lr, input_size):
        super(CriticModel, self).__init__()
        self.lr = lr
        self.h = 256
        self.linear1 = nn.Linear(input_size, self.h)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.h, self.h)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.h, 1)
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


# --- 2. Main Training Function (Corrected Loop) ---
def main():
    # --- CHANGED: Updated run name ---
    wandb.init(project="cacla-vs-cleanrl-benchmark", name="cacla")
    
    # --- CHANGED: Added render_mode="rgb_array" for video ---
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # --- ADDED: Wrap the env for video recording ---
    # This will save a video every 50 episodes to a "videos/" folder
    env = gym.wrappers.RecordVideo(
        env, 
        "videos/cacla-run", 
        episode_trigger=lambda ep_num: ep_num % 50 == 0
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Hyperparameters
    actor_lr = 1e-4
    critic_lr = 3e-4
    gamma = 0.99
    sigma_start = 1.0
    sigma_end = 0.1
    sigma_decay = 500
    
    # --- CHANGED: Set total_timesteps to match PPO/SAC ---
    total_timesteps = 200000 
    
    actor = ActorModel(lr=actor_lr, input_size=state_dim, output_size=action_dim, max_action=max_action)
    critic = CriticModel(lr=critic_lr, input_size=state_dim)
    
    wandb.watch(actor, log="all")
    wandb.watch(critic, log="all")

    print(f"Starting CACLA training on Pendulum-v1 for {total_timesteps} steps...")
    
    global_step = 0
    episode_num = 0
    
    # --- CHANGED: Loop based on global_step ---
    while global_step < total_timesteps:
        state, info = env.reset()
        
        # --- CHANGED: Decay sigma based on episode number ---
        sigma = sigma_end + (sigma_start - sigma_end) * np.exp(-1. * episode_num / sigma_decay)
        
        total_reward = 0
        terminated = False
        truncated = False
        t = 0 # Timestep counter
        
        while not terminated and not truncated:
            # --- ADDED: Check for stopping condition INSIDE the loop ---
            if global_step >= total_timesteps:
                break
                
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action_mean = actor(state_tensor)
            action_explored = torch.distributions.Normal(action_mean, sigma).sample()
            action_clipped = action_explored.clamp(-max_action, max_action)
            
            action_for_env = action_clipped.detach().numpy().flatten()
            next_state, reward, terminated, truncated, info = env.step(action_for_env)
            
            total_reward += reward
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                value_new = critic(next_state_tensor)
            value = critic(state_tensor)
            
            if terminated:
                target = torch.tensor([[reward]], dtype=torch.float32) 
            else:
                target = reward + gamma * value_new
            
            critic.update_weights(prediction=value, target=target)
            
            td_error = target - value
            if td_error.item() > 0.0:
                actor.update_weights(target=action_explored, prediction=action_mean)

            state = next_state
            t += 1
            global_step += 1

        # Log results at the end of an episode
        if episode_num % 10 == 0:
            print(f"Episode: {episode_num}, Global Steps: {global_step}, Reward: {total_reward:.2f}")
        
        wandb.log({
            "episode": episode_num, 
            "charts/episodic_return": total_reward,
            "steps_per_episode": t,
            "sigma": sigma,
            "global_step": global_step 
        })
        
        episode_num += 1

    env.close()
    print(f"Finished training after {global_step} steps.")

if __name__ == "__main__":
    main()