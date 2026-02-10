"""
Enhanced ServiceEnv - With comprehensive error handling and logging
"""
import gymnasium as gym
import numpy as np
import time
import logging
from datetime import datetime
from requests import Session, RequestException, Timeout, ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RobustServiceEnvWrapper(gym.Wrapper):
    """
    Robust wrapper for ServiceEnv

    Features:
    1. Catches all API errors and logs them
    2. Uses last successful values on failure
    3. Automatic retry mechanism
    4. Detailed error logging
    """

    def __init__(self, env, max_retries=3, retry_delay=5.0, error_log_file=None):
        super().__init__(env)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Cache last successful observation
        self.last_successful_obs = None

        # Error statistics
        self.error_count = {
            'get_observation': 0,
            'perform_action': 0,
            'total': 0
        }

        # Error log file
        self.error_log_file = error_log_file
        if error_log_file:
            self._init_error_log()

        # Get underlying ServiceEnv (bypassing gymnasium's private attribute protection)
        self._base_env = self._get_base_env()

    def _get_base_env(self):
        """Get underlying ServiceEnv, bypassing all wrappers"""
        env = self.env
        # Search downward for ServiceEnv
        while hasattr(env, 'env'):
            if hasattr(env, '_perform_action'):  # Found ServiceEnv
                return env
            env = env.env
        # If not found, use unwrapped
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

    def _init_error_log(self):
        """Initialize error log CSV"""
        import csv
        with open(self.error_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "error_type", "operation", "attempt",
                "error_message", "recovery_action", "success"
            ])

    def _log_error(self, error_type, operation, attempt, error_msg, recovery_action, success):
        """Log error to CSV"""
        if not self.error_log_file:
            return

        import csv
        with open(self.error_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                error_type,
                operation,
                attempt,
                str(error_msg).replace('\n', ' ')[:200],  # Limit length
                recovery_action,
                "Yes" if success else "No"
            ])

    def _robust_get_observation(self):
        """
        Robust observation retrieval

        Strategy:
        1. Try to get new observation (up to max_retries attempts)
        2. If failed, use last successful observation
        3. If no cache, return zero vector and warn
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call original environment's _get_observation (using _base_env to bypass gymnasium protection)
                obs = self._base_env._get_observation()

                # Success: cache observation
                self.last_successful_obs = obs

                if attempt > 1:
                    logger.info(f"[OK] Get observation succeeded (attempt {attempt})")
                    self._log_error(
                        "Recovery", "get_observation", attempt,
                        "Previous attempts failed", "Retry succeeded", True
                    )

                return obs

            except Timeout as e:
                error_msg = f"API timeout: {e}"
                logger.warning(f"[WARN] Get observation failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "Timeout", "get_observation", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Use cached", False
                )

            except ConnectionError as e:
                error_msg = f"Connection error: {e}"
                logger.error(f"[ERROR] Get observation failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "ConnectionError", "get_observation", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Use cached", False
                )

            except RequestException as e:
                error_msg = f"Request error: {e}"
                logger.error(f"[ERROR] Get observation failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "RequestError", "get_observation", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Use cached", False
                )

            except Exception as e:
                error_msg = f"Unknown error: {type(e).__name__}: {e}"
                logger.error(f"[ERROR] Get observation failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "UnknownError", "get_observation", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Use cached", False
                )

            # Wait before retry
            if attempt < self.max_retries:
                logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)

        # All retries failed: use cached value
        self.error_count['get_observation'] += 1
        self.error_count['total'] += 1

        if self.last_successful_obs is not None:
            logger.warning(f"[WARN] Using last successful observation (total failures: {self.error_count['get_observation']})")
            self._log_error(
                "Fallback", "get_observation", self.max_retries,
                "All retries failed", "Use last successful observation", True
            )
            return self.last_successful_obs
        else:
            # No cache: return zero vector
            logger.error("[CRITICAL] No cached observation! Returning zero vector")
            self._log_error(
                "Critical", "get_observation", self.max_retries,
                "All retries failed and no cache", "Return zero observation", False
            )
            return np.zeros(self.observation_space.shape)

    def _preprocess_action(self, action):
        """
        Preprocess action format for ServiceEnv compatibility

        Problem: DummyVecEnv adds batch dimension, causing action shape (1, n) instead of (n,)
        Solution: Remove batch dimension and convert to correct format before passing to ServiceEnv

        Args:
            action: Action from PPO/VecEnv, may be array with shape=(1,1)

        Returns:
            processed_action: numpy array with shape=(1,), each element is a scalar
        """
        # Convert to numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Remove batch dimension: (1, n) -> (n,)
        if action.ndim > 1:
            action = action.squeeze()  # Remove all dimensions of size 1

        # Ensure 1D array
        if action.ndim == 0:  # If scalar, convert to 1D array
            action = np.array([action])

        # Process each element in array
        processed = []
        for val in action:
            # If element itself is an array, extract scalar
            if isinstance(val, np.ndarray):
                val = val.item() if val.size == 1 else val[0]

            # Convert to numpy scalar (preserve type)
            if isinstance(val, (float, np.floating)):
                # Check action space type, round if integer range
                if hasattr(self.env, 'action_space'):
                    if hasattr(self.env.action_space, 'dtype') and np.issubdtype(self.env.action_space.dtype, np.integer):
                        val = np.int64(round(val))
                    else:
                        val = np.float64(val)
            elif isinstance(val, (int, np.integer)):
                val = np.int64(val)

            processed.append(val)

        return np.array(processed)

    def _robust_perform_action(self, action):
        """
        Robust action execution

        Strategy:
        1. Preprocess action format (remove batch dimension)
        2. Try to execute action (up to max_retries attempts)
        3. If failed, log warning but don't interrupt training
        4. Training continues, but the action was not actually applied
        """
        # Preprocess action format
        try:
            processed_action = self._preprocess_action(action)
            logger.debug(f"Action preprocessed: {action.shape if hasattr(action, 'shape') else type(action)} -> {processed_action.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Action preprocessing failed: {e}")
            self._log_error(
                "PreprocessError", "preprocess_action", 1,
                str(e), "Use original action", False
            )
            processed_action = action  # Use original action on failure

        for attempt in range(1, self.max_retries + 1):
            try:
                # Call original environment's _perform_action (using _base_env to bypass gymnasium protection)
                self._base_env._perform_action(processed_action)

                if attempt > 1:
                    logger.info(f"[OK] Action execution succeeded (attempt {attempt})")
                    self._log_error(
                        "Recovery", "perform_action", attempt,
                        "Previous attempts failed", "Retry succeeded", True
                    )

                return True  # Success

            except Timeout as e:
                error_msg = f"API timeout: {e}"
                logger.warning(f"[WARN] Action execution failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "Timeout", "perform_action", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Skip action", False
                )

            except ConnectionError as e:
                error_msg = f"Connection error: {e}"
                logger.error(f"[ERROR] Action execution failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "ConnectionError", "perform_action", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Skip action", False
                )

            except RequestException as e:
                error_msg = f"Request error: {e}"
                logger.error(f"[ERROR] Action execution failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "RequestError", "perform_action", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Skip action", False
                )

            except Exception as e:
                error_msg = f"Unknown error: {type(e).__name__}: {e}"
                logger.error(f"[ERROR] Action execution failed (attempt {attempt}/{self.max_retries}): {error_msg}")
                self._log_error(
                    "UnknownError", "perform_action", attempt,
                    error_msg, "Retry" if attempt < self.max_retries else "Skip action", False
                )

            # Wait before retry
            if attempt < self.max_retries:
                logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)

        # All retries failed
        self.error_count['perform_action'] += 1
        self.error_count['total'] += 1

        logger.error(f"[ERROR] Action execution failed, skipping this action (total failures: {self.error_count['perform_action']})")
        self._log_error(
            "Fallback", "perform_action", self.max_retries,
            "All retries failed", "Skip action (no change to system)", False
        )
        return False  # Failure

    def reset(self, **kwargs):
        """Reset environment (with error handling)"""
        logger.info("Resetting environment...")
        obs = self._robust_get_observation()
        self._base_env._last_observation = obs
        return obs, {}

    def step(self, action):
        """
        Execute one step (with error handling)

        Flow:
        1. Execute action (may fail)
        2. Wait
        3. Get new observation (may fail, use cache)
        4. Calculate reward
        5. Check termination condition
        """
        # 1. Execute action
        action_success = self._robust_perform_action(action)

        # 2. Wait (using _base_env to bypass gymnasium protection)
        time.sleep(self._base_env._t)

        # 3. Get observation
        next_obs = self._robust_get_observation()

        # 4. Calculate reward (using _base_env to bypass gymnasium protection)
        try:
            reward = self._base_env._reward.calculate_reward(self._base_env._last_observation, next_obs)
        except Exception as e:
            logger.error(f"[ERROR] Calculate reward failed: {e}")
            self._log_error(
                "RewardError", "calculate_reward", 1,
                str(e), "Use zero reward", False
            )
            reward = 0.0

        # 5. Check termination condition (using _base_env to bypass gymnasium protection)
        try:
            finished, truncated = self._base_env._finisher.terminate_or_truncate(next_obs, reward)
        except Exception as e:
            logger.error(f"[ERROR] Check termination condition failed: {e}")
            self._log_error(
                "FinisherError", "terminate_or_truncate", 1,
                str(e), "Continue episode", False
            )
            finished, truncated = False, False

        # Update last_observation (using _base_env to bypass gymnasium protection)
        self._base_env._last_observation = next_obs

        # Return additional info
        info = {
            'action_success': action_success,
            'error_count': self.error_count.copy()
        }

        return next_obs, reward, finished, truncated, info

    def get_error_summary(self):
        """Get error statistics summary"""
        return {
            'total_errors': self.error_count['total'],
            'get_observation_errors': self.error_count['get_observation'],
            'perform_action_errors': self.error_count['perform_action']
        }
