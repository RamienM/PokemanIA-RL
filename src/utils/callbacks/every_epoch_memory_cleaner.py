from stable_baselines3.common.callbacks import BaseCallback
import gc
import torch

class EveryEpochMemoryCleaner(BaseCallback):
    """
    Callback that performs memory cleanup at the end of training.
    It explicitly calls garbage collection and clears CUDA cache if available,
    helping to free up system and GPU memory.
    """

    def __init__(self):
        """
        Initializes the callback.
        """
        super().__init__()

    def _on_step(self):
        """
        This callback does not perform any action every step,
        so just returns True to continue training.
        """
        return True
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        Performs garbage collection and frees CUDA cache if GPU is available.
        """
        print("[MemoryCleaner] Releasing memory...")
        gc.collect()  # Run Python garbage collector to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory cache
        print("[MemoryCleaner] Memory released.")
