import random

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
            if self.rng.random() < FAILURE_PROB[self.state]: # how to simulate a probability
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

if __name__ == "__main__":

    env = MaintenanceEnv()

    # Test 1: Reset
    state, info = env.reset(seed=11)
    assert state == HEALTHY
    assert env.steps == 0
    print("Test reset: OK")

    # Test 2: Production from Worn
    env.reset(seed=11)
    env.state = WORN

    state, reward, terminated, truncated, info = env.step(PRODUCE)

    print("Production:", state, reward, info)

    assert state == WORN
    assert reward == REWARD_SUCCESS
    assert info["produced"] == 1
    print("Test production: OK")

    # Test 3: Maintenance from Critical
    env.reset(seed=11)
    env.state = CRITICAL

    state, reward, terminated, truncated, info = env.step(MAINTAIN)

    assert state == HEALTHY
    assert reward == REWARD_MAINTENANCE
    assert info["maintenance"] is True
    print("Test maintenance: OK")

    # Test 4: Episode truncation
    env.reset(seed=11)

    for _ in range(MAX_STEPS):
        state, reward, terminated, truncated, info = env.step(MAINTAIN)

    assert truncated is True
    assert terminated is False
    assert env.steps == MAX_STEPS
    print("Test truncation: OK")