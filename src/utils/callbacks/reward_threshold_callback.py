from stable_baselines3.common.callbacks import BaseCallback

class RewardThresholdCallback(BaseCallback):
    """
    Callback that stops training if the total episode reward
    falls below a defined threshold, and optionally saves the model.
    """

    def __init__(self, reward_threshold=-100, verbose=0):
        """
        Initializes the callback.
        
        :param reward_threshold: The reward threshold below which training stops.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        Checks the episode reward info to determine if training should stop.
        
        :return: False to stop training if threshold crossed, True to continue otherwise.
        """
        infos = self.locals.get("infos", [])

        # Check each info dict for episode completion and reward
        for info in infos:
            if "episode" in info:
                total_reward = info["episode"]["r"]

                if total_reward <= self.reward_threshold:
                    # Stop training if total reward is below or equal threshold
                    return False

        return True  # Continue training otherwise
