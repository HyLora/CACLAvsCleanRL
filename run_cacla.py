import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import wandb
import time 
from collections import deque

# --- CONFIGURATION ---
# CACLA is an older algorithm, so it often uses simple hyperparameters.
# It relies heavily on "Sigma" (Noise) for exploration since the Actor is deterministic.

# --- MODELS ---

# The Actor: Decides WHAT to do.
# unlike PPO/SAC, this Actor is DETERMINISTIC. It outputs a single specific number (action).
# It does not output a probability distribution.
class ActorModel(nn.Module):
    def __init__(self, lr, input_size, output_size, max_action):
        super(ActorModel, self).__init__()
        self.lr = lr
        self.h = 256
        self.max_action = max_action
        
        # Simple Feed-Forward Network
        self.linear1 = nn.Linear(input_size, self.h); self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.h, self.h); self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.h, output_size); self.tanh = nn.Tanh()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # We use MSE Loss because CACLA treats RL like Supervised Learning
        self.loss_fnc = nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x); x = self.relu1(x); x = self.linear2(x); x = self.relu2(x)
        x = self.linear3(x); x = self.tanh(x)
        # Scale output to environment limits (e.g. -2 to 2)
        x = x * self.max_action
        return x

    def update_weights(self, prediction, target):
        # Move the network output (prediction) closer to the action we actually took (target)
        loss = self.loss_fnc(prediction, target.detach()) 
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

# The Critic: Decides HOW GOOD a state is.
# This is a standard Value Function V(s).
class CriticModel(nn.Module):
    def __init__(self, lr, input_size):
        super(CriticModel, self).__init__()
        self.lr = lr; self.h = 256
        self.linear1 = nn.Linear(input_size, self.h); self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.h, self.h); self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.h, 1) # Outputs a single scalar Value
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 
        self.loss_fnc = nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x); x = self.relu1(x); x = self.linear2(x); x = self.relu2(x)
        x = self.linear3(x)
        return x

    def update_weights(self, prediction, target):
        loss = self.loss_fnc(prediction, target.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

# --- MAIN ---
def main():
    # Initialize Weights & Biases
    wandb.init(
        project="cacla-vs-cleanrl-benchmark", 
        name="cacla",
        settings=wandb.Settings(init_timeout=300)
    )
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # Optional: Record video every 100 episodes
    env = gym.wrappers.RecordVideo(env, "videos/cacla-run", episode_trigger=lambda ep_num: ep_num % 100 == 0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize Agents
    actor = ActorModel(lr=1e-4, input_size=state_dim, output_size=action_dim, max_action=max_action)
    critic = CriticModel(lr=3e-4, input_size=state_dim)
    
    wandb.watch(actor, log="all"); wandb.watch(critic, log="all")

    # --- GOAL SETTINGS ---
    TARGET_REWARD = -150.0
    REWARD_WINDOW_SIZE = 20
    reward_window = deque(maxlen=REWARD_WINDOW_SIZE)
    
    total_timesteps = 1000000 
    
    # Exploration Noise Settings (Sigma)
    # Since the Actor is deterministic, we MUST add manual noise to explore.
    # We decay this noise over time (start wild, end precise).
    sigma_start = 1.0; sigma_end = 0.1; sigma_decay = 100000
    
    global_step = 0; episode_num = 0
    start_time = time.time()
    
    print(f"--- CACLA Started on Pendulum-v1 ---")
    
    while global_step < total_timesteps:
        state, info = env.reset()
        
        # Calculate current noise level
        sigma = sigma_end + (sigma_start - sigma_end) * np.exp(-1. * episode_num / sigma_decay)
        
        total_reward = 0; terminated = False; truncated = False; t = 0
        
        while not terminated and not truncated:
            if global_step >= total_timesteps: break
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # 1. Get Deterministic Action
            action_mean = actor(state_tensor)
            
            # 2. Add Gaussian Noise for Exploration
            action_explored = torch.distributions.Normal(action_mean, sigma).sample()
            
            # 3. Clip action to environment bounds
            action_clipped = action_explored.clamp(-max_action, max_action)
            
            # 4. Step Environment
            next_state, reward, terminated, truncated, info = env.step(action_clipped.detach().numpy().flatten())
            total_reward += reward
            
            # 5. Critic Update (Standard TD Learning)
            # Target = Reward + Gamma * Value_Next
            with torch.no_grad(): 
                value_new = critic(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
            
            value = critic(state_tensor)
            
            if terminated:
                target = torch.tensor([[reward]], dtype=torch.float32)
            else:
                target = reward + 0.99 * value_new
            
            critic.update_weights(prediction=value, target=target)
            
            # 6. CACLA ACTOR UPDATE (The Unique Part)
            # We calculate the Advantage: (Target - Value)
            # This represents: "Was the actual outcome better than what we expected?"
            td_error = (target - value).item()
            
            if td_error > 0.0: 
                # POSITIVE SURPRISE:
                # The action we took (action_explored) resulted in a better outcome than the Critic predicted.
                # So, we update the Actor to output something closer to `action_explored` next time.
                actor.update_weights(target=action_explored, prediction=action_mean)
            
            # NEGATIVE SURPRISE:
            # If td_error <= 0, the action was worse than expected (or average).
            # CACLA simply ignores this data. It only learns from success.

            state = next_state; t += 1; global_step += 1

        # --- CHECK FOR SUCCESS ---
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)
        
        if episode_num % 10 == 0: 
            print(f"Ep: {episode_num}, Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")

        wandb.log({
            "episode": episode_num, 
            "charts/episodic_return": total_reward, 
            "charts/average_return": avg_reward, 
            "global_step": global_step
        })
        
        if len(reward_window) == REWARD_WINDOW_SIZE and avg_reward >= TARGET_REWARD:
             elapsed_time = time.time() - start_time
             print("\n" + "="*40)
             print(f"🚀 CACLA SOLVED THE ENVIRONMENT!")
             print(f"🏆 Episodes: {episode_num}")
             print(f"⏱️ Time Taken: {elapsed_time:.2f} seconds")
             print(f"📈 Avg Reward: {avg_reward:.2f}")
             print("="*40 + "\n")
             # exit(0) # Uncomment to stop immediately
        
        episode_num += 1
    env.close()

if __name__ == "__main__": main()