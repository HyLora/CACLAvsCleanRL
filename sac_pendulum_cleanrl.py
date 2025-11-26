import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import time
from collections import deque

# ================= CONFIGURATION =================
# These hyperparameters control how the AI learns.
ENV_ID = "Pendulum-v1"
TOTAL_STEPS = 200_000       # Total frames to interact with the environment
BUFFER_SIZE = 1_000_000     # SAC is "Off-Policy", so it needs a large memory to store past experiences
BATCH_SIZE = 256            # How many experiences to learn from at once
GAMMA = 0.99                # Discount factor: How much we care about future rewards vs immediate ones
TAU = 0.005                 # Soft Update rate: How fast the "Target" networks copy the main networks
LR = 3e-4                   # Learning Rate: How big of a step to take during Gradient Descent
ALPHA = 0.2                 # Entropy Coefficient: The "Temperature". High = Explore more (Random), Low = Exploit more (Greedy)
START_STEPS = 5000          # Steps to take completely randomly at the start to fill the buffer
POLICY_DELAY = 2            # Update the Actor less frequently than the Critic for stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SOLVED CRITERIA ---
TARGET_REWARD = -150.0   # If average reward > -150, we consider the pendulum "balanced"
REWARD_WINDOW = 20       # We need to maintain this score for 20 consecutive episodes

# ================= W&B INIT =================
# Initialize Weights & Biases for live graphing
wandb.init(
    project="cacla-vs-cleanrl-benchmark",
    name="sac-explained",
    config={
        "env": ENV_ID,
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "lr": LR,
        "alpha": ALPHA
    },
    settings=wandb.Settings(init_timeout=300)
)

# ================= REPLAY BUFFER =================
# Stores transitions (State, Action, Reward, Next_State, Done).
# Since SAC is Off-Policy, it can learn from data collected minutes or hours ago.
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly pick a batch of experiences
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ================= NETWORKS =================

# --- The Actor (Policy Network) ---
# Goal: Output the best action (mean) and how uncertain it is (std) for a given state.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        
        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Output 1: The Mean action (mu)
        self.mean = nn.Linear(256, action_dim)
        # Output 2: The Log Standard Deviation (sigma). We predict log(std) for numerical stability.
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        # Clamp log_std to prevent the distribution from becoming too narrow (collapse) or too wide (instability)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        
        # Create a Normal (Gaussian) distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization Trick (rsample):
        # Allows gradients to flow through the stochastic sampling process.
        # Ideally: action = mean + std * noise
        x_t = normal.rsample()
        
        # Tanh Squashing:
        # Neural networks output values from -inf to +inf.
        # Physical robots/simulations have limits (e.g., -2 to +2).
        # Tanh forces the output to be between -1 and 1.
        y_t = torch.tanh(x_t)
        
        # Scale to environment limits (Pendulum is -2 to 2)
        action = y_t * self.max_action
        
        # Log Probability Correction:
        # When we squash a Gaussian with Tanh, the probability density changes.
        # We must subtract a correction term to get the true log_prob of the squashed action.
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        
        # Sum log_probs across action dimensions (if action_dim > 1)
        return action, log_prob.sum(-1, keepdim=True)

# --- The Critic (Q-Network) ---
# Goal: Estimate "How good is this State-Action pair?"
# SAC uses "Double Q-Learning" (Two Critics) to prevent overestimating values.
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Critic 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        
        # Critic 2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Concatenate State and Action because Q(s, a) depends on both
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

# ================= MAIN TRAINING LOOP =================
env = gym.make(ENV_ID)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0]) # Pendulum max_action is 2.0

# Initialize Models
actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
critic = Critic(state_dim, action_dim).to(DEVICE)

# Target Critic: A stable copy of the critic used to calculate future targets.
# This stops the "moving target" problem in recursive learning.
target_critic = Critic(state_dim, action_dim).to(DEVICE)
target_critic.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=LR)
critic_opt = optim.Adam(critic.parameters(), lr=LR)

buffer = ReplayBuffer(BUFFER_SIZE)

# Tracking variables
state, _ = env.reset()
episode_reward = 0
episode_num = 0
global_step = 0
reward_history = deque(maxlen=REWARD_WINDOW)
start_time = time.time()

print(f"--- SAC Started on {DEVICE} ---")
print(f"Goal: Average reward > {TARGET_REWARD} over {REWARD_WINDOW} episodes")

# === THE LOOP ===
for step in range(TOTAL_STEPS):
    global_step += 1

    # 1. Action Selection
    if step < START_STEPS:
        # Warmup Phase: Take purely random actions to fill buffer with diverse data
        action = env.action_space.sample()
    else:
        # Training Phase: Use the Actor to select actions
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = actor.sample(state_tensor)
        action = action.cpu().numpy()[0] # Convert back to numpy for Gym

    # 2. Environment Step
    next_state, reward, done, truncated, _ = env.step(action)
    
    # 3. Store in Buffer
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    # 4. Handle Episode End
    if done or truncated:
        episode_num += 1
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)

        # Log to WandB
        wandb.log({
            "episode": episode_num,
            "charts/episodic_return": episode_reward,
            "charts/average_return": avg_reward,
            "global_step": global_step
        })
        
        print(f"Ep {episode_num}: Reward={episode_reward:.2f} | Avg={avg_reward:.2f} | Step={global_step}")

        # Check for Solution
        if len(reward_history) == REWARD_WINDOW and avg_reward >= TARGET_REWARD:
            elapsed_time = time.time() - start_time
            print("\n" + "="*40)
            print(f"🚀 ENVIRONMENT SOLVED!")
            print(f"🏆 Episodes: {episode_num}")
            print(f"⏱️ Time Taken: {elapsed_time:.2f} seconds")
            print(f"📈 Avg Reward: {avg_reward:.2f}")
            print("="*40 + "\n")
            exit(0) # Uncomment to stop training when solved

        state, _ = env.reset()
        episode_reward = 0

    # 5. Training Step (Only if buffer has enough data)
    if len(buffer) > BATCH_SIZE:
        # Retrieve a batch of data
        s, a, r, s2, d = buffer.sample(BATCH_SIZE)
        
        # Convert to PyTorch tensors
        s = torch.tensor(s, dtype=torch.float32).to(DEVICE)
        a = torch.tensor(a, dtype=torch.float32).to(DEVICE)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        s2 = torch.tensor(s2, dtype=torch.float32).to(DEVICE)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # --- A. Update Critic ---
        with torch.no_grad():
            # Get next action from Actor for the NEXT state
            next_action, next_log_prob = actor.sample(s2)
            
            # Get Q-values from Target Critics
            q1_next, q2_next = target_critic(s2, next_action)
            
            # Take the minimum Q-value (prevents overestimation bias)
            min_q_next = torch.min(q1_next, q2_next)
            
            # ENTROPY TERM: Subtract (alpha * log_prob). 
            # This rewards the agent for exploring (high entropy/randomness) alongside getting high rewards.
            target_q = r + (1 - d) * GAMMA * (min_q_next - ALPHA * next_log_prob)

        # Get current Q-values
        q1, q2 = critic(s, a)
        
        # Calculate Loss (MSE) for both critics
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        # Optimize Critic
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # --- B. Update Actor (Delayed) ---
        # We update the actor less frequently than the critic (e.g., every 2 steps)
        # This gives the Critic time to stabilize before the Actor tries to exploit it.
        if global_step % POLICY_DELAY == 0:
            # We want to Maximize (Q - alpha * log_prob)
            # Since optimizers minimize loss, we take the negative: -(Q - alpha * log_prob)
            
            new_action, log_prob = actor.sample(s)
            q1_new, q2_new = critic(s, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (ALPHA * log_prob - q_new).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # --- C. Soft Update Target Networks ---
            # Slowly move Target Network weights towards Main Network weights
            # theta_target = tau * theta_main + (1 - tau) * theta_target
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

print("Training finished.")
env.close()