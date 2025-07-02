import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

def merge_dicts(dicts):
    """
    Merges a list of dictionaries by computing the mean of numeric values for each key,
    and also returns the distribution of values per key as numpy arrays.

    :param dicts: List of dictionaries to merge.
    :return: Tuple (mean_dict, distrib_dict) where
             mean_dict contains the mean values per key,
             distrib_dict contains the numpy arrays of values per key.
    """
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict


class TensorboardCallback(BaseCallback):
    """
    Callback adapted from "https://github.com/PWhiddy/PokemonRedExperiments/blob/master/v2/tensorboard_callback.py"
    Logs agent statistics from multiple parallel environments to TensorBoard, including
    scalar summaries and histograms of distributions. Also saves exploration maps at training end.
    """

    def __init__(self, log_dir, num_envs, verbose=0):
        """
        Initializes the TensorBoard callback.

        :param log_dir: Directory path to save TensorBoard logs.
        :param num_envs: Number of parallel environments.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.num_envs = num_envs
        self.step_counters = [0] * num_envs

    def _on_training_start(self):
        """
        Creates a TensorBoard SummaryWriter at the start of training if not already created.
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        Retrieves agent stats from each environment, merges them,
        logs scalar mean values and histograms of distributions to TensorBoard.

        :return: True to continue training.
        """
        envs = self.model.get_env()
        dicts = []

        # Collect stats dictionaries from each environment
        for env_idx in range(self.num_envs):
            dicts.append(envs.env_method("get_agent_stats", indices=env_idx)[0])

        # Compute means and distributions for each metric key
        mean_infos, distributions = merge_dicts(dicts)

        # Log mean scalar values and distributions as histograms
        for key, val in mean_infos.items():
            self.logger.record(f"env_stats/{key}", val)
            self.writer.add_scalar(f"env_stats/{key}", val, self.num_timesteps)

        for key, distrib in distributions.items():
            self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.num_timesteps)
            self.logger.record(f"env_stats_max/{key}", max(distrib))

        return True

    def _on_training_end(self):
        """
        Called at the end of training.
        Retrieves visit_count_maps from all environments,
        converts them into exploration maps, and logs them as images.
        Closes the TensorBoard writer.
        """
        # Retrieve visit count maps from all parallel environments
        all_visit_maps = self.training_env.get_attr("visit_count_map")

        # Convert visit_count_maps to binary exploration maps (255 = explored, 0 = unexplored)
        explore_maps_derived = np.array([(vm > 0).astype(np.uint8) * 255 for vm in all_visit_maps])

        # Aggregate exploration maps using max over parallel envs to create consolidated map
        map_sum = reduce(explore_maps_derived, "f h w -> h w", "max")
        self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

        # Arrange exploration maps in a grid for visualization
        if explore_maps_derived.shape[0] == 1:
            # Single environment: remove the 'f' dimension
            map_row = explore_maps_derived.squeeze(0)
        else:
            # Multiple environments: arrange maps in a 2-row grid for display
            map_row = rearrange(explore_maps_derived, "(r f) h w -> (r h) (f w)", r=2)

        self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

        # Flush logger to ensure all logs are written
        self.logger.dump(self.n_calls)

        # Close the TensorBoard writer to free resources
        if self.writer:
            self.writer.close()
