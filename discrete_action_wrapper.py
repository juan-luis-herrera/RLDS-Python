"""
Discrete Action Space Wrapper - Solves PPO exploration deficiency problem
"""
import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.Wrapper):
    """
    Converts integer-range Box action space to Discrete space

    Problem:
    - Box(low=0, high=16, dtype=int) causes PPO to output continuous values (e.g., -0.017)
    - PPO's output concentrates around 0, unable to fully explore the action range

    Solution:
    - Convert to Discrete(17), forcing PPO to output discrete action indices
    - Mapping: 0->0, 1->1, 2->2, ..., 16->16
    """

    def __init__(self, env):
        super().__init__(env)

        # Check original action space
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DiscreteActionWrapper only supports Box action space")

        # Get original range
        self.original_low = int(env.action_space.low[0])
        self.original_high = int(env.action_space.high[0])
        self.action_range = self.original_high - self.original_low + 1

        # Create discrete action space
        self.action_space = gym.spaces.Discrete(self.action_range)

        print(f"\n{'='*60}")
        print(f"DiscreteActionWrapper activated")
        print(f"{'='*60}")
        print(f"Original action space: Box({self.original_low}, {self.original_high})")
        print(f"New action space: Discrete({self.action_range})")
        print(f"Mapping:")
        for i in range(self.action_range):
            real_value = self._discrete_to_continuous(i)
            print(f"  Action index {i} -> EncodingThreadCount = {real_value[0]}")
        print(f"{'='*60}\n")

    def _discrete_to_continuous(self, discrete_action):
        """
        Convert discrete action index to actual EncodingThreadCount value

        Args:
            discrete_action: Integer from 0 to (action_range-1)

        Returns:
            Actual configuration value (0 to 16)
        """
        if isinstance(discrete_action, np.ndarray):
            discrete_action = discrete_action.item()

        continuous_value = self.original_low + int(discrete_action)
        return np.array([continuous_value], dtype=np.int64)

    def step(self, action):
        """
        Convert discrete action to continuous value, then pass to original environment

        Args:
            action: Discrete action from PPO (integer 0-16)

        Returns:
            Standard gym step return values
        """
        # Convert action format
        if isinstance(action, np.ndarray):
            if action.ndim > 1:
                action = action.squeeze()
            if action.ndim == 0:
                action = action.item()
            elif action.ndim == 1:
                action = action[0]

        # Discrete -> Continuous
        continuous_action = self._discrete_to_continuous(action)

        # Call original environment
        return self.env.step(continuous_action)

    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)
