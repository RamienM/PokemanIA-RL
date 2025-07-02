from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import torch

class PPOAgent:
    def __init__(self, env, policy="MlpPolicy", verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, tensorboard_log=None):
        """
        Initialize the PPO model with the given parameters.

        :param env: The environment to train the model on.
        :param policy: The policy architecture to use (default 'MlpPolicy').
        :param verbose: Verbosity level (default 1).
        :param n_steps: Number of steps per update (default 2048).
        :param batch_size: Batch size for each update (default 64).
        :param n_epochs: Number of epochs per update (default 10).
        :param gamma: Discount factor (default 0.99).
        :param ent_coef: Entropy coefficient (default 0.01).
        :param tensorboard_log: Directory for TensorBoard logs (default None).
        """
        # Set device to GPU if available, else CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[PPO] Using device: {device}")
        
        # Initialize the PPO model with given parameters and device
        self.model = PPO(
            policy, env, verbose=verbose, n_steps=n_steps, batch_size=batch_size, 
            n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, tensorboard_log=tensorboard_log, device=device
        )

    def train(self, total_timesteps, callbacks=None, tb_log_name="poke_ppo", progress_bar=False):
        """
        Train the PPO agent.

        :param total_timesteps: Total number of training timesteps.
        :param callbacks: Optional list of callbacks for logging, monitoring, etc.
        :param tb_log_name: TensorBoard log name.
        :param progress_bar: Whether to show a progress bar during training.
        """
        # Ensure callbacks is a list if provided
        if callbacks is not None and not isinstance(callbacks, list):
            callbacks = [callbacks]

        # Wrap callbacks in CallbackList if any callbacks are given
        callback_list = CallbackList(callbacks) if callbacks else None

        # Start learning process with provided parameters and callbacks
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
            progress_bar=progress_bar
        )

    def save(self, path="models/ppo/ppo_agent"):
        """
        Save the PPO model to the specified path.

        :param path: File path where the model will be saved.
        """
        self.model.save(path)

    def load(self, path="models/ppo/ppo_agent", env=None):
        """
        Load a PPO model from the specified path.

        :param path: File path from where the model will be loaded.
        :param env: Optional environment to associate with the loaded model.
        """
        self.model = PPO.load(path, env)

    def predict(self, obs, deterministic=True):
        """
        Predict an action given an observation using the PPO model.

        :param obs: Observation from the environment.
        :param deterministic: Whether to return deterministic actions (default True).
        :return: Predicted action and state.
        """
        return self.model.predict(obs, deterministic=deterministic)
