import random
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import json

# States
HEALTHY = 0
WORN = 1
CRITICAL = 2

# Actions
PRODUCE = 0
MAINTAIN = 1

# Transition probabilities
FAILURE_PROB = (0.01, 0.07, 0.38)
DEGRADATION_PROB = (0.17, 0.28, 0.0)

# Rewards
REWARD_SUCCESS = 5.0
REWARD_MAINTENANCE = -3.5
REWARD_FAILURE = -18.0

# Episode duration
MAX_STEPS = 60

N_STATES = 3
N_ACTIONS = 2

# Training parameters
TRAIN_EPISODES = 2500
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY_FRACTION = 0.70

class MaintenanceEnv:
    """Simulate a robotic cell with stochastic wear and failures"""

    def __init__(self):
        """Initialize the environment"""
        self.rng = random.Random()
        self.state = HEALTHY
        self.steps = 0

    def reset(self, seed=None):
        """Reset the environment to its initial state"""
        if seed is not None:
            self.rng.seed(seed)

        self.state = HEALTHY
        self.steps = 0

        return self.state, {}

    def step(self, action):
        """Execute an action and update the environment state"""

        if action not in (PRODUCE, MAINTAIN):
            raise ValueError("Invalid action")

        if self.steps >= MAX_STEPS:
            raise RuntimeError("Episode finished: call reset()")

        self.steps += 1

        info = {
            "failure": False,
            "produced": 0,
            "maintenance": False
        }

        if action == MAINTAIN:
            self.state = HEALTHY
            reward = REWARD_MAINTENANCE
            info["maintenance"] = True

        else:
            # Check if a failure occurs
            if self.rng.random() < FAILURE_PROB[self.state]:
                self.state = HEALTHY
                reward = REWARD_FAILURE
                info["failure"] = True

            else:
                # Successful production
                reward = REWARD_SUCCESS
                info["produced"] = 1

                # Check if degradation occurs
                if self.rng.random() < DEGRADATION_PROB[self.state]:
                    self.state = min(self.state + 1, CRITICAL)

        terminated = False
        truncated = self.steps >= MAX_STEPS

        return self.state, reward, terminated, truncated, info

class QLearningAgent:
    """Represent a tabular Q-learning agent"""

    def __init__(self, alpha=0.12, gamma=0.95, seed=None):
        """Initialize Q-values, learning parameters, and random generator"""
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

        self.q_table = [
            [0.0 for _ in range(N_ACTIONS)]
            for _ in range(N_STATES)
        ]

    def update(self, state, action, reward, next_state):
        """Update a Q-value using temporal-difference learning"""

        current_q = self.q_table[state][action]

        # Estimate the best future Q-value
        best_next_q = max(self.q_table[next_state])

        # Compute the temporal-difference target and error
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q

        # Update the Q-table
        self.q_table[state][action] += self.alpha * td_error

    def choose_action(self, state, epsilon):
        """Select an action using an epsilon-greedy policy"""

        # Exploration
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)

        # Exploitation
        if self.q_table[state][PRODUCE] >= self.q_table[state][MAINTAIN]:
            return PRODUCE
        else:
            return MAINTAIN

def choose_baseline_action(strategy, step, rng):
    """Select an action using a fixed baseline strategy"""

    if strategy == "always_produce":
        return PRODUCE

    elif strategy == "periodic_8":
        if (step + 1) % 8 == 0:
            return MAINTAIN
        return PRODUCE

    elif strategy == "random":
        return rng.randrange(N_ACTIONS)

    else:
        raise ValueError("Invalid baseline strategy")

def train_episode(env, agent, epsilon):
    """Train the agent for one episode"""

    state, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:

        # Select an action
        action = agent.choose_action(state, epsilon)

        # Interact with the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the Q-table
        agent.update(state, action, reward, next_state)

        # Accumulate reward
        total_reward += reward

        # Move to the next state
        state = next_state

        # Check episode completion
        done = terminated or truncated

    return total_reward

def train_agent(env, agent, n_episodes=TRAIN_EPISODES):
    """Train the agent over multiple episodes with epsilon decay"""

    rewards = []
    epsilons = []

    decay_episodes = max(1, int(n_episodes * EPSILON_DECAY_FRACTION))

    for episode in range(n_episodes):

        # Compute exponentially decaying epsilon
        progress = min(episode / decay_episodes, 1.0)

        epsilon = EPSILON_START * (
            EPSILON_MIN / EPSILON_START
        ) ** progress

        # Train for one episode
        episode_reward = train_episode(env, agent, epsilon)

        # Store training results
        rewards.append(episode_reward)
        epsilons.append(epsilon)

    return rewards, epsilons

def train_multiple_runs(seeds=(11, 23, 37)):
    """Train independent agents using different random seeds"""

    runs = []

    for seed in seeds:

        # Create a new environment and agent
        env = MaintenanceEnv()
        env.reset(seed=seed)

        agent = QLearningAgent(seed=seed)

        # Train the agent
        rewards, epsilons = train_agent(env, agent)

        # Extract the learned greedy policy
        policy = [
            agent.choose_action(state, epsilon=0.0)
            for state in range(N_STATES)
        ]

        # Store the results
        runs.append({
            "seed": seed,
            "agent": agent,
            "rewards": rewards,
            "policy": policy,
            "epsilons": epsilons
        })

        print(f"\nSeed: {seed}")
        print(f"Last 100 mean reward: {sum(rewards[-100:])/100:.2f}")
        print(f"Learned policy: {policy}")

    return runs

def evaluate_agent(agent, n_episodes=300, baseline=None):
    """Evaluate a trained agent or a fixed baseline"""

    env = MaintenanceEnv()

    total_reward = 0.0
    total_failures = 0
    total_maintenance = 0
    total_produced = 0
    failure_free_episodes = 0

    for episode in range(n_episodes):

        # Use new reproducible test scenarios
        state, _ = env.reset(seed=900000 + episode)

        # Initialize an independent random generator
        baseline_rng = random.Random(1000000 + episode)

        done = False
        episode_failures = 0

        while not done:

            # Select the action based on the evaluation strategy
            if baseline is None:
                action = agent.choose_action(state, epsilon=0.0)
            else:
                action = choose_baseline_action(
                    baseline, env.steps, baseline_rng
                )

            # Execute the action without learning
            next_state, reward, terminated, truncated, info = env.step(action)

            # Collect performance metrics
            total_reward += reward
            total_failures += int(info["failure"])
            total_maintenance += int(info["maintenance"])
            total_produced += info["produced"]

            episode_failures += int(info["failure"])

            state = next_state
            done = terminated or truncated

        if episode_failures == 0:
            failure_free_episodes += 1

    return {
        "mean_reward": total_reward / n_episodes,
        "mean_failures": total_failures / n_episodes,
        "mean_maintenance": total_maintenance / n_episodes,
        "mean_produced": total_produced / n_episodes,
        "failure_free_rate": failure_free_episodes / n_episodes
    }

def print_comparison(runs, baseline_results):
    """Compare Q-learning and baseline evaluation results"""

    metrics = [
        "mean_reward",
        "mean_failures",
        "mean_maintenance",
        "mean_produced",
        "failure_free_rate"
    ]

    # Compute average results across trained agents
    q_results = {}

    for metric in metrics:
        values = [run["evaluation"][metric] for run in runs]
        q_results[metric] = sum(values) / len(values)

    # Combine Q-learning and baseline results
    strategies = {
        "Always Produce": baseline_results["always_produce"],
        "Periodic 8": baseline_results["periodic_8"],
        "Random": baseline_results["random"],
        "Q-learning": q_results
    }

    # Print comparison table
    print("\nStrategy comparison:")
    print(f"{'Strategy':<18} {'Reward':>8} {'Failures':>9} {'Maintenance':>12} {'Produced':>9}")

    for name, result in strategies.items():
        print(
            f"{name:<18}"
            f"{result['mean_reward']:>8.2f}"
            f"{result['mean_failures']:>9.2f}"
            f"{result['mean_maintenance']:>12.2f}"
            f"{result['mean_produced']:>9.2f}"
        )

    return strategies

def save_evaluation_summary(strategies):
    """Save evaluation metrics to a CSV file"""

    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "evaluation_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        # Write column headers
        writer.writerow([
            "Strategy",
            "Mean reward",
            "Mean failures",
            "Mean maintenance",
            "Mean produced",
            "Failure-free rate (%)"
        ])

        # Write evaluation results
        for name, result in strategies.items():
            writer.writerow([
                name,
                result["mean_reward"],
                result["mean_failures"],
                result["mean_maintenance"],
                result["mean_produced"],
                result["failure_free_rate"] * 100
            ])

    return output_path

def save_q_tables(runs):
    """Save learned Q-values and greedy policies to CSV"""

    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "learned_q_tables.csv"

    state_names = ["Healthy", "Worn", "Critical"]

    with output_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Seed",
            "State",
            "Q Produce",
            "Q Maintain",
            "Greedy action"
        ])

        # Save the Q-table of each trained agent
        for run in runs:

            q_table = run["agent"].q_table

            for state in range(N_STATES):

                q_produce = q_table[state][PRODUCE]
                q_maintain = q_table[state][MAINTAIN]

                # Select the action with the highest Q-value
                best_action = (
                    "Produce" if q_produce >= q_maintain
                    else "Maintain"
                )

                writer.writerow([
                    run["seed"],
                    state_names[state],
                    q_produce,
                    q_maintain,
                    best_action
                ])

    return output_path

def save_training_history(runs, window=100):
    """Save training rewards and moving averages to CSV"""

    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "training_history.csv"

    n_episodes = len(runs[0]["rewards"])

    # Compute mean reward across runs
    mean_rewards = []

    for i in range(n_episodes):
        values = [run["rewards"][i] for run in runs]
        mean_rewards.append(sum(values) / len(values))

    with output_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Episode",
            "Epsilon",
            "Reward seed 11",
            "Reward seed 23",
            "Reward seed 37",
            "Mean reward",
            "Moving average"
        ])

        for i in range(n_episodes):

            # Compute moving average
            start = max(0, i - window + 1)

            moving_avg = (
                sum(mean_rewards[start:i + 1]) / (i - start + 1)
            )

            writer.writerow([
                i + 1,
                runs[0]["epsilons"][i],
                runs[0]["rewards"][i],
                runs[1]["rewards"][i],
                runs[2]["rewards"][i],
                mean_rewards[i],
                moving_avg
            ])

    return output_path

def save_configuration(runs):
    """Save environment and training parameters to JSON"""

    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "config.json"

    agent = runs[0]["agent"]

    config = {
        "environment": {
            "states": ["Healthy", "Worn", "Critical"],
            "actions": ["Produce", "Maintain"],
            "failure_probabilities": FAILURE_PROB,
            "degradation_probabilities": DEGRADATION_PROB,
            "reward_success": REWARD_SUCCESS,
            "reward_maintenance": REWARD_MAINTENANCE,
            "reward_failure": REWARD_FAILURE,
            "max_steps": MAX_STEPS
        },
        "training": {
            "algorithm": "Q-learning",
            "episodes": TRAIN_EPISODES,
            "seeds": [run["seed"] for run in runs],
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon_start": EPSILON_START,
            "epsilon_min": EPSILON_MIN,
            "epsilon_decay_fraction": EPSILON_DECAY_FRACTION
        },
        "evaluation": {
            "episodes": 300,
            "seed_start": 900000,
            "epsilon": 0.0
        }
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    return output_path

def plot_learning_curve(runs, window=100):
    """Plot the moving average of training rewards across runs"""

    n_episodes = len(runs[0]["rewards"])

    # Compute mean reward across runs
    mean_rewards = []

    for i in range(n_episodes):
        values = [run["rewards"][i] for run in runs]
        mean_rewards.append(sum(values) / len(values))

    # Compute moving average
    moving_average = []

    for i in range(n_episodes):
        start = max(0, i - window + 1)
        recent_rewards = mean_rewards[start:i + 1]

        moving_average.append(
            sum(recent_rewards) / len(recent_rewards)
        )

    # Create the plot
    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, n_episodes + 1),
        moving_average,
        label=f"Moving average ({window} episodes)"
    )

    plt.xlabel("Training episode")
    plt.ylabel("Mean episode reward")
    plt.title("Q-learning - Training curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "learning_curve.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    return output_path

def plot_strategy_comparison(strategies):
    """Plot mean evaluation rewards for all strategies"""

    names = list(strategies.keys())
    rewards = [strategies[name]["mean_reward"] for name in names]

    # Create the bar chart
    plt.figure(figsize=(9, 5))

    bars = plt.bar(names, rewards)

    plt.bar_label(bars, fmt="%.1f", padding=3)

    plt.xlabel("Strategy")
    plt.ylabel("Mean total reward (60 steps)")
    plt.title("Evaluation - Strategy comparison")
    plt.ylim(0, max(rewards) * 1.15)

    plt.tight_layout()

    # Save the plot
    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "strategy_comparison.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    return output_path

def plot_q_table(agent, seed):
    """Plot the learned Q-values and highlight greedy actions"""

    q_table = agent.q_table

    fig, ax = plt.subplots(figsize=(7, 4))

    # Create the heatmap
    image = ax.imshow(q_table, cmap="Blues", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Produce", "Maintain"])

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Worn", "Critical"])

    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title(f"Learned Q-table - Seed {seed}")

    # Display Q-values and highlight the best action
    for state in range(N_STATES):

        best_action = max(
            range(N_ACTIONS),
            key=lambda action: q_table[state][action]
        )

        for action in range(N_ACTIONS):

            value = q_table[state][action]

            bbox = {
                "boxstyle": "square,pad=0.5",
                "facecolor": "none",
                "edgecolor": "red",
                "linewidth": 2
            } if action == best_action else None

            ax.text(
                action, state, f"{value:.1f}",
                ha="center",
                va="center",
                color="black",
                bbox=bbox
            )

    fig.colorbar(image, ax=ax, label="Q-value")
    plt.tight_layout()

    # Save the plot
    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"q_table_seed_{seed}.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    return output_path

def plot_operational_metrics(strategies):
    """Compare failures and maintenance actions across strategies"""

    names = list(strategies.keys())

    failures = [
        strategies[name]["mean_failures"] for name in names
    ]

    maintenance = [
        strategies[name]["mean_maintenance"] for name in names
    ]

    # Define bar positions
    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot grouped bars
    bars_failure = ax.bar(
        [i - width / 2 for i in x],
        failures,
        width,
        label="Breakdowns"
    )

    bars_maintenance = ax.bar(
        [i + width / 2 for i in x],
        maintenance,
        width,
        label="Maintenance actions"
    )

    # Add labels and formatting
    ax.bar_label(bars_failure, fmt="%.2f", padding=3)
    ax.bar_label(bars_maintenance, fmt="%.2f", padding=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean events per episode (60 steps)")
    ax.set_title("Evaluation - Operational metrics")
    ax.legend()

    plt.tight_layout()

    # Save the plot
    output_dir = Path("risultati_manutenzione")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "operational_metrics.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    return output_path

def test_worn_transitions(n_trials=10000):
    """Verify transition probabilities when producing from Worn"""

    env = MaintenanceEnv()
    env.reset(seed=11)

    counts = {HEALTHY: 0, WORN: 0, CRITICAL: 0}
    total_reward = 0

    for _ in range(n_trials):

        # Reset without reseeding the random generator
        env.reset()

        # Set the initial state to Worn
        env.state = WORN

        # Execute one production step
        next_state, reward, _, _, _ = env.step(PRODUCE)

        counts[next_state] += 1
        total_reward += reward

    # Compute empirical probabilities
    p_failure = counts[HEALTHY] / n_trials
    p_worn = counts[WORN] / n_trials
    p_critical = counts[CRITICAL] / n_trials
    avg_reward = total_reward / n_trials

    print(f"Failure: {p_failure:.2%} (expected 7%)")
    print(f"Worn: {p_worn:.2%} (expected 66.96%)")
    print(f"Critical: {p_critical:.2%} (expected 26.04%)")
    print(f"Average reward: {avg_reward:.2f} (expected 3.39)")

    # Check results within a tolerance
    assert abs(p_failure - 0.07) < 0.015
    assert abs(p_worn - 0.6696) < 0.02
    assert abs(p_critical - 0.2604) < 0.02
    assert abs(avg_reward - 3.39) < 0.3

    print("Test probabilities: OK")

def main():
    """Run the complete maintenance RL experiment"""

    # Train three independent agents
    runs = train_multiple_runs()

    # Evaluate the trained agents
    for run in runs:
        run["evaluation"] = evaluate_agent(run["agent"])

    # Evaluate baseline strategies
    baseline_names = [
        "always_produce",
        "periodic_8",
        "random"
    ]

    baseline_results = {}

    for name in baseline_names:
        baseline_results[name] = evaluate_agent(
            None, baseline=name
        )

    # Compare evaluation results
    comparison = print_comparison(runs, baseline_results)

    # Generate plots
    plot_learning_curve(runs)
    plot_strategy_comparison(comparison)
    plot_q_table(runs[0]["agent"], runs[0]["seed"])
    plot_operational_metrics(comparison)

    # Save experiment results
    save_evaluation_summary(comparison)
    save_q_tables(runs)
    save_training_history(runs)
    save_configuration(runs)

    print("\nExperiment completed successfully")


if __name__ == "__main__":
    main()