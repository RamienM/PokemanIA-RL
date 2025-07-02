# Source imports
from src.environments.Pokemon_Red.env import PokemonRedEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.stream_agent_wrapper import StreamWrapper
from src.utils.video_recorder import VideoRecorder

from src.utils.callbacks.tensorboard_callback import TensorboardCallback
from src.utils.callbacks.every_epoch_memory_cleaner import EveryEpochMemoryCleaner
from src.utils.callbacks.agent_stats_logger_callback import AgentStatsLoggerCallback
from src.utils.callbacks.reward_threshold_callback import RewardThresholdCallback


# Emulator imports
from emulators.pyboy.emulator import GameEmulator
from emulators.pyboy.memory_reader import MemoryReader
from models.segmentation.STELLE_Pokemon_Segmentation.inference.inferencer import STELLEInferencer

#Queue imports
from server.shared_inferencer import SharedInferencer
from server.inference_server import inference_process


#Stable baselines 3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

#Weight & Bias imports
import wandb
from wandb.integration.sb3 import WandbCallback

#Others imports
import yaml
import os
import sys
import json
from datetime import datetime
import multiprocessing as mp

def make_env(rank,session_path, config,  request_q, response_q, seed=0):
    """
    Utility function to create a multiprocess environment.
    
    Args:
        rank (int): Index of the subprocess.
        session_path (str): Path for saving videos and logs.
        config (dict): Configuration dictionary.
        request_q (multiprocessing.Queue): Queue to send inference requests.
        response_q (multiprocessing.Queue): Queue to receive inference responses.
        seed (int): Seed for RNG to ensure reproducibility.
    
    Returns:
        Callable that initializes and returns the environment instance.
    """
    def _init():
        # Initialize emulator and memory reader
        emulator = GameEmulator(config)
        memory_reader = MemoryReader(emulator)
        
        # Choose shared vision model if server mode enabled, otherwise create own model
        if config.get("server", False):
            print(f"[ENV - {rank}] Using shared vision model")
            vision_model = SharedInferencer(request_q, response_q)
        else:
            print(f"[ENV - {rank}] Using own vision model")
            vision_model = STELLEInferencer()
        
        # Optionally set up video recording
        video_recorder = None
        if config.get("save_video", False):
            save_path = os.path.join(session_path, "videos", f"env_{rank}")
            os.makedirs(save_path, exist_ok=True)
            video_recorder = VideoRecorder(save_path=save_path)
        
        # Create environment wrapped with streaming metadata for monitoring/logging
        env = StreamWrapper(
            PokemonRedEnv(emulator, memory_reader, vision_model, video_recorder, config), 
            stream_metadata={
                "user": "Ramien",
                "env_id": rank,
                "color": "#7A378B",
                "extra": "STELLE",
                "sprite_id": 4
            }
        )
        
        # Reset environment with unique seed for each subprocess
        env.reset(seed=(seed + rank))
        return env
    
    # Set random seed for reproducibility
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    # Set multiprocessing start method (spawn is safer)
    mp.set_start_method("spawn", force=True) 

    # Load configuration from YAML file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Number of CPUs/environments to run in parallel
    num_cpu = config.get("num_cpu", 1)

    # Checkpoint file to load, priority: command-line argument > config file
    checkpoint_from_yaml =config.get("load_checkpoint", "")
    checkpoint_arg = sys.argv[1] if len(sys.argv) > 1 else None
    file_name = checkpoint_arg if checkpoint_arg else checkpoint_from_yaml

    # Training parameters derived from config and number of CPUs
    ep_length = config.get("max_steps", 1000000) * num_cpu
    train_steps_batch = config.get("train_steps_batch", ep_length // 1024) 
    batch_size = config.get("batch_size", ep_length//256)

    iterations = config.get("iterations", 1)
    progress_bar = config.get("progress_bar",False)

    server_enabled = config.get("server", False)
    use_wandb_logging = config.get("use_wandb", True)

    
    # Setup session directory and timestamp for saving results and logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sess_id = f"poke_{timestamp}"

    root_dir = config.get("root_dir","game&results/Pokemon_red_env/checkpoints")
    base_dir = os.path.join(root_dir, sess_id)
    sess_path_logs = os.path.join(base_dir, "tensorboard")  # logs para TensorBoard
    agent_stats_dir = os.path.join(base_dir, "agent_stats")  # CSVs

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(sess_path_logs, exist_ok=True)
    os.makedirs(agent_stats_dir, exist_ok=True)

    # Start inference server and queues if server mode enabled
    if server_enabled:
        manager = mp.Manager()
        request_q = manager.Queue()
        
        response_queues = [manager.Queue() for _ in range(num_cpu)]
        inference_proc = mp.Process(
            target=inference_process,
            args=(request_q,),
        )
        inference_proc.start()
        env = SubprocVecEnv([make_env(i, base_dir, config,  request_q, response_queues[i]) for i in range(num_cpu)])
    else:
        env = SubprocVecEnv([make_env(i,base_dir,  config,  None, None) for i in range(num_cpu)])

    # Save metadata about the training run
    metadata = {
        "session_id": sess_id,
        "checkpoint_freq": config.get("checkpoint_save_freq", 64),
        "num_envs": num_cpu,
        "ep_length": ep_length,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "notes": config.get("notes", "")  # si agregas esto a tu config.yaml
    }

    metadata_path = os.path.join(base_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Prepare callbacks for checkpoints, logging, cleaning, and reward monitoring
    callbacks = [
        CheckpointCallback(save_freq=config.get("checkpoint_save_freq", 64),save_path=base_dir,name_prefix=sess_id), 
        TensorboardCallback(sess_path_logs,num_cpu), 
        EveryEpochMemoryCleaner(),
        AgentStatsLoggerCallback(base_dir=agent_stats_dir,num_envs=num_cpu,save_every=config.get("save_stats_every",50)),
        RewardThresholdCallback(reward_threshold=config.get("reward_threshold",-100))
    ]

    # Initialize Weights & Biases logging if enabled
    if use_wandb_logging:
        wandb.tensorboard.patch(root_logdir=str(sess_path_logs))
        run = wandb.init(
            project=config.get("wandb_project", "pokemon-train"),
            id=sess_id,
            name=sess_id,
            config=config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
            reinit=True
        )
        callbacks.append(WandbCallback())


    # Initialize PPO agent with specified hyperparameters
    model = PPOAgent(
        env=env,
        policy="MultiInputPolicy",
        verbose=1,
        n_steps=train_steps_batch,
        batch_size=batch_size,
        n_epochs=1,
        gamma=0.997,
        ent_coef=0.01,
        tensorboard_log=sess_path_logs
    )

    # Load checkpoint if available
    if os.path.exists(file_name + ".zip"):
        print("[TRAIN] Loading checkpoint...")
        model.load(path=file_name, env=env)
        print(f"[TRAIN] Loaded checkpoint {file_name}.zip")

    # Start training
    print("[TRAIN] Starting training")
    model.train(total_timesteps=ep_length, callbacks=CallbackList(callbacks), progress_bar=progress_bar)
    print("[TRAIN] Training finished")
        
    # Cleanup after training
    env.close()

    # Save the final model checkpoint
    last_checkpoint_path = os.path.join(base_dir, "final")
    model.save(last_checkpoint_path)

    # Save the path of the last checkpoint for future reference
    with open("last_checkpoint_path.txt", "w") as f:
        f.write(last_checkpoint_path)

    # Signal inference server process to stop if running
    if server_enabled:
        request_q.put((None, None, None)) 
        inference_proc.join()

    # Finish wandb run
    if use_wandb_logging:
        run.finish()