# CACLA vs CleanRL Benchmark

This repository contains implementations of three Reinforcement Learning algorithms trained on the Pendulum-v1 environment. It benchmarks a custom implementation of CACLA against standard PPO and SAC implementations adapted from CleanRL.

## Project Structure

* run_cacla.py: Custom implementation of the Continuous Actor-Critic Learning Automaton (CACLA) algorithm.
* ppo_pendulum_cleanrl.py: Proximal Policy Optimization (PPO) implementation, adapted from CleanRL benchmarks.
* sac_pendulum_cleanrl.py: Soft Actor-Critic (SAC) implementation, adapted from CleanRL benchmarks.
* run_all.sh: A shell script to execute all three algorithms in parallel for benchmarking.

## Dependencies

The project requires Python 3.10+ and the following libraries:

    pip install gymnasium[classic_control] torch numpy wandb

## How to Run

### 1. Run a Single Algorithm
You can run any algorithm individually using Python:

    # Run CACLA
    python run_cacla.py

    # Run PPO
    python ppo_pendulum_cleanrl.py

    # Run SAC
    python sac_pendulum_cleanrl.py

### 2. Run the Full Benchmark
To run all three algorithms simultaneously (as background processes), use the provided shell script:

    # Make the script executable (only needed once)
    chmod +x run_all.sh

    # Run the benchmark
    ./run_all.sh

## Logging and Visualization

All scripts are configured to log training metrics to Weights & Biases (WandB) under the project name "cacla-vs-cleanrl-benchmark".

* Metric Tracking: Episodic return, average return, and algorithm-specific metrics (e.g., sigma, entropy).
* Video Recording: Videos of the agent are saved locally in the "videos/" folder (e.g., videos/cacla, videos/Pendulum-v1/ppo).
