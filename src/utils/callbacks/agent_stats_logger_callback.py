from stable_baselines3.common.callbacks import BaseCallback
import os
import json

class AgentStatsLoggerCallback(BaseCallback):
    """
    Callback that saves the agent_stats[] from each environment thread into separate JSONL files,
    writing data every N steps.
    """

    def __init__(self, base_dir="agent_stats_logs", num_envs=1, save_every=1, verbose=0):
        """
        Initializes the callback.

        :param base_dir: Base directory to save logs.
        :param num_envs: Number of parallel environments.
        :param save_every: Number of steps between each save.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.base_dir = base_dir
        self.num_envs = num_envs
        self.save_every = save_every
        self.files = []  # File handlers for each environment's log file
        self.step_counters = [0] * num_envs  # Step counters for each environment

    def _on_training_start(self) -> None:
        """
        Called when training starts. 
        Creates directories and opens files for logging agent stats for each environment.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        for i in range(self.num_envs):
            env_dir = os.path.join(self.base_dir, f"env_{i}")
            os.makedirs(env_dir, exist_ok=True)
            jsonl_path = os.path.join(env_dir, "agent_stats.jsonl")

            f = open(jsonl_path, "a", encoding="utf-8")
            self.files.append(f)

    def _on_step(self) -> bool:
        """
        Called at every step of the training loop.
        Collects agent stats from each environment and writes them to the corresponding file
        every `save_every` steps.

        :return: True to continue training.
        """
        envs = self.model.get_env()

        for env_idx in range(self.num_envs):
            self.step_counters[env_idx] += 1

            if self.step_counters[env_idx] >= self.save_every:
                # Retrieve the agent stats from the environment method "get_agent_stats"
                stats_list = envs.env_method("get_agent_stats", indices=env_idx)[0]
                if stats_list:
                    f = self.files[env_idx]
                    json.dump(stats_list, f, default=str)  # Serialize stats to JSON
                    f.write("\n")  # Write a newline to separate JSON entries
                
                # Reset the step counter after saving
                self.step_counters[env_idx] = 0

        return True

    def _on_training_end(self) -> None:
        """
        Called when training ends.
        Closes all opened log files.
        """
        for f in self.files:
            f.close()
