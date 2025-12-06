import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import wandb
import time 
import random
from collections import deque

# --- CONFIGURATION ---
class ActorModel(nn.Module):
    def __init__(self, lr, input_size, output_size, max_action):
        super(ActorModel, self).__init__()
        self.max_action = max_action
        self.h = 128 # Smaller network often works better for simple Pendulum
        
        self.net = nn.Sequential(
            nn.Linear(input_size, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, output_size),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x) * self.max_action

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CriticModel(nn.Module):
    def __init__(self, lr, input_size):
        super(CriticModel, self).__init__()
        self.h = 128
        
        self.net = nn.Sequential(
            nn.Linear(input_size, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fnc = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def update(self, prediction, target):
        loss = self.loss_fnc(prediction, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(np.array(action), dtype=torch.float32),
            torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(np.array(terminated), dtype=torch.float32).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

# --- MAIN ---
def main():
    wandb.init(
        project="cacla-vs-cleanrl-benchmark", 
        name="cacla-robust-batch",
        settings=wandb.Settings(init_timeout=300)
    )
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Lower Learning Rates + Batching = Stability
    actor = ActorModel(lr=3e-4, input_size=state_dim, output_size=action_dim, max_action=max_action)
    critic = CriticModel(lr=1e-3, input_size=state_dim)
    
    buffer = ReplayBuffer(capacity=20000)
    BATCH_SIZE = 64
    
    TARGET_REWARD = -200.0 # Strict success threshold
    REWARD_WINDOW_SIZE = 20
    reward_window = deque(maxlen=REWARD_WINDOW_SIZE)
    
    total_timesteps = 1000000 
    
    # Exploration Settings
    sigma_start = 2.0
    sigma_end = 0.1
    sigma_decay_steps = 10000 # Decay over global steps, not episodes
    
    global_step = 0; episode_num = 0
    start_time = time.time()
    
    print(f"--- CACLA (Robust) Started ---")
    
    while global_step < total_timesteps:
        state, info = env.reset()
        total_reward = 0; terminated = False; truncated = False
        
        while not (terminated or truncated):
            # Calculate Sigma based on global steps
            sigma = sigma_end + (sigma_start - sigma_end) * np.exp(-1. * global_step / 3000)
            
            # 1. Action Selection
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_mean = actor(state_tensor)
                
                # Explore
                noise = torch.randn_like(action_mean) * sigma
                action = (action_mean + noise).clamp(-max_action, max_action)
            
            action_np = action.cpu().numpy().flatten()
            
            # 2. Step
            next_state, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            
            # Store experience
            buffer.push(state, action_np, reward / 10.0, next_state, terminated)
            
            state = next_state
            global_step += 1
            
            # 3. Training Step (Only if we have enough data)
            if len(buffer) > BATCH_SIZE:
                b_state, b_action, b_reward, b_next_state, b_terminated = buffer.sample(BATCH_SIZE)
                
                # --- Critic Update ---
                with torch.no_grad():
                    value_next = critic(b_next_state)
                    # CRITICAL FIX: Only mask if terminated (death), NOT truncated (time limit)
                    target_value = b_reward + 0.99 * value_next * (1 - b_terminated)
                
                value_current = critic(b_state)
                critic.update(value_current, target_value)
                
                # --- Actor Update (CACLA Logic) ---
                # We re-evaluate the collected actions against the current actor's prediction
                # In standard CACLA, we check if the collected action had a positive TD error
                
                # Calculate TD error for the batch
                td_error = target_value - value_current
                
                # Identify which samples were "good" (positive surprise)
                # We detach td_error because we don't backprop through the boolean check
                good_samples_idx = (td_error > 0).squeeze()
                
                if good_samples_idx.sum() > 0:
                    # We want the actor to move towards the actions that resulted in high value
                    # The 'target' for the actor is the action from the buffer that caused the good surprise
                    actions_to_imitate = b_action[good_samples_idx]
                    states_for_update = b_state[good_samples_idx]
                    
                    current_predictions = actor(states_for_update)
                    
                    # MSE Loss between current prediction and the "good" action
                    actor_loss = nn.MSELoss()(current_predictions, actions_to_imitate)
                    actor.update(actor_loss)

        # --- LOGGING ---
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)
        
        if episode_num % 10 == 0: 
            sps = int(global_step / (time.time() - start_time))
            print(f"Ep: {episode_num}, Reward: {total_reward:.1f}, Avg: {avg_reward:.1f}, Sigma: {sigma:.2f}, SPS: {sps}")

        wandb.log({
            "charts/episodic_return": total_reward, 
            "charts/average_return": avg_reward, 
            "charts/sigma": sigma,
            "global_step": global_step
        })
        
        if len(reward_window) == REWARD_WINDOW_SIZE and avg_reward >= TARGET_REWARD:
             print(f"🚀 SOLVED! Avg Reward: {avg_reward:.2f}")
             exit(0)
        
        episode_num += 1

if __name__ == "__main__": main()