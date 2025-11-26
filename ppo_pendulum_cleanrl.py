import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import wandb 
from gymnasium import ObservationWrapper
import time
from collections import deque

# ================= CONFIGURATION =================
ENV_ID = "Pendulum-v1"
TOTAL_TIMESTEPS = 1_000_000 # PPO is "On-Policy" (less efficient), so it needs more steps than SAC
NUM_STEPS = 2048            # How many steps to collect before stopping to learn (The "Rollout")
TARGET_REWARD = -150.0      # Solved threshold
REWARD_WINDOW = 20          # Stability window

# ================= WRAPPERS =================
# Reinforcement Learning is very sensitive to the scale of input numbers.
# These wrappers ensure the Neural Network sees nice, normalized numbers (mean 0, std 1).
def _clip_obs_recursive(obs, clip):
    if isinstance(obs, np.ndarray): return np.clip(obs, -clip, clip)
    if isinstance(obs, (list, tuple)): return type(obs)(_clip_obs_recursive(o, clip) for o in obs)
    if isinstance(obs, dict): return {k: _clip_obs_recursive(v, clip) for k, v in obs.items()}
    return obs

class ClipObservationWrapper(ObservationWrapper):
    def __init__(self, env, clip=10.0):
        super().__init__(env)
        self.clip = float(clip)
        self.observation_space = env.observation_space 
    def observation(self, observation):
        return _clip_obs_recursive(observation, self.clip)

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        # Note: We track rewards manually in the loop, but these wrappers help the Agent learn.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = ClipObservationWrapper(env, 10.0)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed); env.observation_space.seed(seed)
        return env
    return thunk

# Orthogonal Initialization: A trick to help deep networks train faster.
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ================= AGENT (ACTOR-CRITIC) =================
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # CRITIC: Estimates Value V(s) --> "How good is this state?"
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), 
            nn.Tanh(), 
            layer_init(nn.Linear(64, 64)), 
            nn.Tanh(), 
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        # ACTOR: Estimates the Mean (mu) of the action.
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), 
            nn.Tanh(), 
            layer_init(nn.Linear(64, 64)), 
            nn.Tanh(), 
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        )
        # Learnable Log Standard Deviation. (Allows the agent to learn how much to explore).
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x): 
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create a Normal (Gaussian) distribution
        probs = Normal(action_mean, action_std)
        
        # If we are playing, sample an action. If we are training, we pass the old action in to get its new probability.
        if action is None: 
            action = probs.sample()
            
        # Return: Action, Log Probability (needed for PPO math), Entropy (randomness), Value
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ================= MAIN LOOP =================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("✅ STARTING PPO - VERSION 3 (FINAL FIX)")
    print("="*50 + "\n")

    wandb.init(
        project="cacla-vs-cleanrl-benchmark", 
        name="ppo-manual-track", 
        monitor_gym=False,
        settings=wandb.Settings(init_timeout=300)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 1)])
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    # STORAGE BUFFERS (PPO collects data, learns, then deletes it)
    obs = torch.zeros((NUM_STEPS, 1) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, 1) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, 1)).to(device)
    rewards = torch.zeros((NUM_STEPS, 1)).to(device)
    dones = torch.zeros((NUM_STEPS, 1)).to(device)
    values = torch.zeros((NUM_STEPS, 1)).to(device)

    global_step = 0
    episode_num = 0
    
    next_obs, _ = envs.reset(seed=1)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    
    num_updates = TOTAL_TIMESTEPS // NUM_STEPS
    
    current_ep_reward = 0
    reward_window = deque(maxlen=REWARD_WINDOW)
    start_time = time.time()

    print(f"--- PPO STARTING ON {device} ---")

    # === OUTER LOOP: EPISODES / UPDATES ===
    for update in range(1, num_updates + 1):
        # Learning Rate Annealing: Slowly lower LR as training progresses
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * 3e-4

        # === PHASE 1: DATA COLLECTION (ROLLOUT) ===
        for step in range(0, NUM_STEPS):
            global_step += 1
            obs[step] = next_obs; dones[step] = next_done
            
            # 1. Get Action from Policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action; logprobs[step] = logprob
            
            # 2. Execute Step
            real_next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            current_ep_reward += reward[0] 
            
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(real_next_obs).to(device), torch.Tensor(done).to(device)

            # 3. Logging & Solved Check
            if done[0]: 
                episode_num += 1
                reward_window.append(current_ep_reward)
                avg_reward = np.mean(reward_window)
                print(f"Ep {episode_num}: Reward={current_ep_reward:.2f} | Avg={avg_reward:.2f} | Step={global_step}")
                wandb.log({"episode": episode_num, "charts/episodic_return": current_ep_reward, "charts/average_return": avg_reward, "global_step": global_step})
                current_ep_reward = 0 
                
                if len(reward_window) == REWARD_WINDOW and avg_reward >= TARGET_REWARD:
                    print(f"\n🚀 PPO SOLVED! Time: {time.time() - start_time:.2f}s")
                    exit(0)

        # === PHASE 2: CALCULATE ADVANTAGE (GAE) ===
        # GAE (Generalized Advantage Estimation) is a smart way to calculate rewards.
        # It balances "Short term actual reward" vs "Long term predicted value".
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1: nextnonterminal = 1.0 - next_done; nextvalues = next_value
                else: nextnonterminal = 1.0 - dones[t + 1]; nextvalues = values[t + 1]
                
                # TD Error: (Reward + Value_Next) - Value_Current
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                
                # Recursive Magic: combines current error with previous error
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the buffer (batch size = NUM_STEPS)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(NUM_STEPS)
        
        # === PHASE 3: OPTIMIZATION (LEARNING) ===
        # We re-train on the data we just collected for 10 epochs
        for epoch in range(10):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, 64):
                end = start + 64; mb_inds = b_inds[start:end]
                
                # Re-evaluate the action using the *current* network
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # Calculate Ratio (New Prob / Old Prob)
                # We use logs because it's numerically safer: log(a/b) = log(a) - log(b)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Normalize Advantages (Technical trick for stability)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # --- THE PPO CLIP LOSS (The core of PPO) --- 
                # 1. Normal Loss: Advantage * Ratio
                pg_loss1 = -mb_advantages * ratio
                # 2. Clipped Loss: Advantage * Ratio (clipped to be close to 1, e.g., 0.8 to 1.2)
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                # Take the max (which is min because of negative sign) to be pessimistic/safe
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss: Train critic to predict better
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy Loss: Bonus for randomness (prevents getting stuck early)
                ent_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - 0.0 * ent_loss + v_loss * 0.5

                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()
        
        print(f"🔄 PPO Update Complete (Step {global_step})")