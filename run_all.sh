#!/bin/bash

echo "Starting benchmark..."
echo "This will run PPO, SAC, and CACLA in parallel."
echo "You can monitor their progress on WandB."

# Run PPO in the background
python ppo_pendulum_cleanrl.py &

# Run the NEW Early-Stopping SAC in the background
python sac_pendulum_cleanrl.py &

# Run the 200k-step CACLA script in the background
# (This is the one we fixed to stop at 200k steps, not 2000 episodes)
python run_cacla.py &

echo "All three scripts are now running."
wait
echo "All benchmarks finished."