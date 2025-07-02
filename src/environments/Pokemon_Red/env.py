from src.environments.Pokemon_Red.global_map import local_to_global, GLOBAL_MAP_SHAPE

import uuid
from gymnasium import Env, spaces
from collections import deque
import numpy as np
import json 
import cv2

class PokemonRedEnv(Env):
    """
    Custom Gymnasium Environment for Pokemon Red.
    This environment simulates playing Pokemon Red, providing observations
    from the game state and calculating rewards based on various in-game metrics.
    """

    def __init__(self, emulator, memory_reader, vision_model, video_recorder, config):
        """
        Initializes the PokemonRedEnv.

        Args:
            emulator: An object to interact with the game emulator (e.g., performing steps, loading states).
            memory_reader: An object to read specific memory addresses from the game.
            vision_model: A model for processing screen images (e.g., for segmentation).
            video_recorder: An object to record video of the gameplay.
            config (dict): A dictionary containing configuration parameters for the environment.
        """
        super().__init__()
        self.emulator = emulator
        self.memory_reader = memory_reader
        self.vision_model = vision_model
        self.video_recorder = video_recorder
        

        ##--------- Config -----------------
        # General environment configuration
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"] # Not directly used in the provided snippet but good to keep
        self.init_state = config["init_state"] # Initial game state file
        self.act_freq = config["action_freq"] # How often actions are performed (not directly used here for step)
        
        # Observation space related configurations
        self.enc_freqs = (
            5 if "fourier_encode_freq" not in config else config["fourier_encode_freq"]
        ) # Not directly used in this snippet
        self.quantity_action_storage = (
            5 if "actions_stack" not in config else config["actions_stack"]
        ) # Number of recent actions to store
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        ) # Scale factor for rewards
        self.max_steps = (
            1000 if "max_steps" not in config else config["max_steps"]
        ) # Maximum number of steps per episode
        self.init_state = (
            "../states/STELLE.state" if "init_state" not in config else config["init_state"]
        ) # Default initial state if not provided
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        ) # Unique identifier for the environment instance
        self.coords_pad = (
            12 if "coords_pad" not in config else config["coords_pad"]
        ) # Padding for the visit map crop

        self.step_discount = config.get("step_discount", 0) # Reward discount per step

        #--------- Observation space config -----------------
        # Deque to store recent actions for observation
        self.recent_actions = deque(maxlen=self.quantity_action_storage)
        # Define the action space as discrete, corresponding to valid emulator actions
        self.action_space = spaces.Discrete(len(self.emulator.VALID_ACTIONS))

        # Define the shape of the main screen and segmented screen observations
        # (height, width, channels/stack depth)
        self.output_shape_main = (72, 80, self.quantity_action_storage)
        # Define the observation space as a dictionary of various game elements
        self.observation_space = spaces.Dict(
            {
                "main_screen": spaces.Box(low=0, high=255, shape=self.output_shape_main, dtype=np.uint8),
                "segmented_screen" : spaces.Box(low=0, high=15, shape=self.output_shape_main, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1), # Player's health fraction
                "badges": spaces.Discrete(8), # Number of badges obtained (max 8)
                "events": spaces.MultiBinary(self.memory_reader.get_difference_between_events()), # Binary flags for events
                "visit_map": spaces.Box(low=0.0, high=1.0, shape=(self.output_shape_main[0], self.output_shape_main[1], 1), dtype=np.float32), # Normalized visit count map
                "recent_actions": spaces.MultiDiscrete([len(self.emulator.VALID_ACTIONS)] * self.quantity_action_storage), # Stack of recent actions
                "remaining_ratio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # Ratio of unexplored cells in current region
                "in_combat" : spaces.MultiBinary(1), # Binary flag indicating if in combat
            }
        )

        self.reset_count = 0 # Counter for environment resets

        #--------- Video recorder -----------------
        if self.video_recorder:
            self.video_recorder.start_writing()

        #---------Loading regions -----------------
        # Load map region data from a JSON file
        regions_path = "src/environments/Pokemon_Red/map_data.json"
        with open(regions_path, "r") as f:
            data = json.load(f)
        self.regions = []
        # Parse region data into a more accessible format
        for r in data["regions"]:
            x0, y0 = r["coordinates"]
            w, h = r["tileSize"]
            # Each region: [x_min, x_max, y_min, y_max, id, name]
            self.regions.append({
                "id": r["id"],
                "name": r["name"],
                "xmin": x0,
                "xmax": x0 + w,
                "ymin": y0,
                "ymax": y0 + h,
            })
        
        #---------Loading events -----------------
        # Load event descriptions from a JSON file
        with open("src/environments/Pokemon_Red/events.json") as f:
            event_names = json.load(f)
        self.event_descriptions_by_addr_bit  = event_names


    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed) # Call parent class reset
        self.seed = seed
        self.emulator.load_state(self.init_state) # Load the initial game state

        # Initialize tracking variables for exploration and progress
        self.seen_coords = {} # Dictionary to store visited coordinates and their visit counts

        # Map to keep track of visit counts for global coordinates
        self.visit_count_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)

        # Stacks to store recent screen frames
        self.recent_main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        self.recent_seg_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        
        # Initialize recent actions deque with zeros
        self.recent_actions = deque([0] * self.quantity_action_storage, maxlen=self.quantity_action_storage)

        # Initialize reward-related variables
        self.last_health = 1 # Last recorded health fraction
        self.total_healing_rew = 0 # Total healing reward accumulated
        self.died_count = 0 # Number of times the player's Pokémon fainted
        self.has_obtained_first_pokemon = False # Flag to track if the first Pokémon has been obtained
        self.party_size = 0 # Current number of Pokémon in the player's party
        self.last_party_size = 0 # Number of Pokémon in the party in the previous step
        self.step_count = 0 # Counter for steps in the current episode
        self.region_count_r = 0 # Reward for entering new regions
        self.stuck = 0 # Not directly used in the snippet, but could be for stuck detection

        self.is_in_battle = False # Flag to indicate if the player is in a battle
        self.batges_in_possesion = 0 # Number of badges obtained
        self.x, self.y, self.m = 0, 0, 0 # Player's coordinates (x, y) and map ID (m)

        # Event tracking
        self.base_event_flags = self.memory_reader.read_events_done() # Events done at reset
        self.all_event_flags_set_history = set() # Set of all event flags that have been activated
        self.current_new_events = [] # List of new events activated in the current step
        self.last_recorded_event_description = [] # Description of the last new event(s)

        # Region exploration tracking
        self.visited_regions = set() # Set of unique region IDs visited
        self.region_cells_seen = {}  # id_region -> set of (x, y) coordinates seen within that region

        # Initialize region cell tracking for all regions
        for region in self.regions:
            self.region_cells_seen[region["id"]] = set()
        
        # Reward for region exploration (per region)
        self.reward_region_exploration = [0] * len(self.regions)

        # Calculate initial progress reward
        self.progress_reward = self.calculate_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {} # Return initial observation and empty info dict
        
    def step(self, action):
        """
        Executes one step in the environment.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing the new observation, reward, 'terminated' flag,
                   'truncated' flag, and an info dictionary.
        """
        self.emulator.step(action) # Perform the action in the emulator
        self.update_recent_actions(action) # Update the deque of recent actions

        # Get current game state information
        self.x, self.y, self.m = self.memory_reader.get_game_coords() # Player's coordinates and map ID
        self.is_in_battle = self.memory_reader.is_in_battle() # Check if in battle
        self.batges_in_possesion = self.memory_reader.read_bagdes_in_possesion() # Number of badges
        self.party_size = self.memory_reader.read_pokemon_in_party() # Number of Pokémon in party

        # Update various internal states and reward components
        self.update_seen_coords() # Track visited coordinates
        self.update_heal_reward() # Update healing reward and death count
        self.update_visit_map() # Update the global visit count map
        self.update_events() # Check for newly activated events

        new_reward = self.update_reward() # Calculate and update the total reward
        step_limit_reached = self.check_if_done() # Check if the episode is done (e.g., max steps reached)
        obs = self._get_obs() # Get the new observation
        self.print_info() # Print progress information if enabled

        self.last_party_size = self.party_size # Update last party size for next step
        self.step_count += 1 # Increment step counter
        info = {"episode": {"r": self.total_reward,  "l": self.step_count, "exploration_reward":self.reward_scale * (self.get_exploration_reward()) * 3}}
        # Return observation, new reward, terminated (always False in this snippet for now),
        # truncated (step limit reached), and info dictionary
        return obs, new_reward, False, step_limit_reached, info

    def _get_obs(self):
        """
        Generates the current observation for the agent.

        Returns:
            dict: A dictionary containing various observation components.
        """
        screen = self.emulator.get_screen() # Get the raw screen image from the emulator
        segmentation = self.segmented_screen(screen) # Get the segmented screen from the vision model

        # Reduce the resolution of the screen and segmentation
        reduced_screen = self.reduce_screen(screen[:, :, 0]) # Take one channel for main screen
        reduced_segmentation = self.reduce_screen(segmentation)

        # Update the recent screen stacks with the new reduced frames
        self.update_recent_stack(self.recent_main_screen, reduced_screen)
        self.update_recent_stack(self.recent_seg_screen, reduced_segmentation)

        # Construct the observation dictionary
        observation = {
            "main_screen": self.recent_main_screen, 
            "segmented_screen": self.recent_seg_screen,
            "health": np.array([self.read_hp_fraction()]), # Current HP fraction
            "badges": self.batges_in_possesion, # Number of badges
            "events": np.array(self.memory_reader.read_event_bits(), dtype=np.int8), # Raw event bits
            "recent_actions": self.recent_actions, # Stack of recent actions
            "remaining_ratio":  np.array([self.get_remaining_in_current_region()], dtype=np.float32), # Remaining unexplored ratio in current region
            "visit_map": self.get_visit_map_crop(),  # Normalized and cropped visit map
            "in_combat": np.array([int(self.is_in_battle)], dtype=np.uint8) # Is in combat (binary)
        }
        
        return observation
    
    def check_if_done(self):
        """
        Checks if the episode should terminate.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        return self.step_count >= self.max_steps - 1

    def print_info(self):
        """
        Prints current reward and step information if `print_rewards` is enabled.
        """
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.5f}"
            prog_string += f" sum: {self.total_reward:5.5f}"
            print(f"\r{prog_string}", end="", flush=True)
    
    def update_recent_actions(self, action):
        """
        Adds the most recent action to the deque of recent actions.

        Args:
            action (int): The action taken in the current step.
        """
        self.recent_actions.appendleft(action)
    
    def update_events(self):
        """
        Updates the set of activated event flags and identifies newly activated events.
        It also stores the descriptions of the new events.
        """
        current_active_addr_bit_flags = set() # Set to store currently active event flags
        
        raw_event_bits = self.memory_reader.read_event_bits() # Read raw event bits from memory
        event_flags_start_addr, event_flags_end_addr = self.memory_reader.get_event_flags() # Get event flag address range

        total_addresses = event_flags_end_addr - event_flags_start_addr
        expected_total_bits = total_addresses * 8

        # Debugging check for unexpected length of event bits
        if len(raw_event_bits) != expected_total_bits:
            print(f"DEBUG: Mismatch in expected event bit length. Expected {expected_total_bits}, got {len(raw_event_bits)}. Processing partial data.")

        bit_counter = 0
        # Iterate through memory addresses and their bits to identify active flags
        for address in range(event_flags_start_addr, event_flags_end_addr):
            for bit_idx_in_byte in range(8):
                if bit_counter < len(raw_event_bits):
                    if raw_event_bits[bit_counter] == 1: # If the bit is set (event is active)
                        key = f"0x{address:X}-{bit_idx_in_byte}" # Form a unique key for the flag
                        current_active_addr_bit_flags.add(key)
                    bit_counter += 1
                else: # Stop if all raw event bits have been processed
                    break 
            if bit_counter >= len(raw_event_bits): # Stop if all raw event bits have been processed
                break 

        # Identify newly activated flags by comparing with historical flags
        newly_activated_flags = current_active_addr_bit_flags - self.all_event_flags_set_history
        
        self.current_new_events = [] # Reset list of new events for this step
        # Populate current_new_events with descriptions of newly activated flags
        for flag_key in newly_activated_flags:
            if flag_key in self.event_descriptions_by_addr_bit:
                self.current_new_events.append(self.event_descriptions_by_addr_bit[flag_key])
            else:
                self.current_new_events.append(f"Unknown Event: {flag_key}")
                print(f"DEBUG: Unknown event key in newly_activated_flags: {flag_key}")
        
        # Update the historical set of all activated event flags
        self.all_event_flags_set_history.update(current_active_addr_bit_flags)

        if self.current_new_events: # If new events were found in this step
            self.last_recorded_event_description = list(self.current_new_events) # Store them as last recorded events
    
    def update_recent_stack(self, stack, new_frame):
        """
        Updates a stack of recent frames (e.g., screen images).
        The oldest frame is shifted out, and the new frame is added to the front.

        Args:
            stack (np.array): The current stack of frames.
            new_frame (np.array): The new frame to add to the stack.
        """
        stack[:, :, 1:] = stack[:, :, :-1]  # Shift existing frames to the right (older)
        stack[:, :, 0] = new_frame # Place the new frame at the front (most recent)
    
    def update_seen_coords(self):
        """
        Records visited coordinates, incrementing their visit count.
        Coordinates are only tracked when not in battle.
        """
        if not self.is_in_battle: # Only track coordinates if not in combat
            coord_string = f"x:{self.x} y:{self.y} m:{self.m}" # Create a unique string for the coordinate
            
            # Increment the visit count for the coordinate
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
            
    
    def read_hp_fraction(self):
        """
        Calculates the fraction of total current HP to total maximum HP for all Pokémon in the party.

        Returns:
            float: The HP fraction (between 0 and 1). Returns 0 if max_hp_sum is 0 to avoid division by zero.
        """
        hp_sum =  self.memory_reader.get_sum_all_current_hp() # Sum of current HP of all Pokémon
        max_hp_sum = self.memory_reader.get_sum_all_max_hp() # Sum of max HP of all Pokémon

        if max_hp_sum == 0:
            return hp_sum # Should ideally be 0 if no Pokémon or max HP is 0
        
        return hp_sum / max_hp_sum
    
    def get_agent_stats(self):
        """
        Gathers and returns a dictionary of various agent statistics for monitoring.

        Returns:
            dict: A dictionary containing current agent statistics.
        """
        levels = self.memory_reader.get_all_player_pokemon_level() # Levels of all Pokémon in party
        reg = self.get_current_region() # Current map region
        region_id   = reg["id"]   if reg else None
        region_name = reg["name"] if reg else "Unknown"
        region_reward = self.get_region_reward() # Reward for region exploration

        return {
                "step": self.step_count, # Current step count
                "x": self.x, # Player's x-coordinate
                "y": self.y, # Player's y-coordinate
                "region_id": region_id, # ID of the current region
                "region_name": region_name, # Name of the current region
                "region_reward": region_reward, # Accumulated region reward
                "pcount": self.party_size, # Number of Pokémon in party
                "team_p": self.memory_reader.get_all_player_pokemon_name(), # Names of Pokémon in party
                "levels": levels, # List of Pokémon levels
                "levels_sum": sum(levels), # Sum of all Pokémon levels
                "hp": self.read_hp_fraction(), # Current HP fraction
                "coord_count": len(self.seen_coords), # Number of unique coordinates visited
                "explore": self.get_exploration_reward(), # Exploration reward
                "deaths": self.died_count, # Number of deaths
                "badge": self.batges_in_possesion, # Number of badges
                "event": self.progress_reward["event"], # Reward from events
                "last_event_done": self.last_recorded_event_description, # Description of last events
                "healr": self.total_healing_rew, # Total healing reward
                "action": self.recent_actions[0], # Most recent action
                "in_combat": self.is_in_battle, # Whether in combat
                "total_reward" : self.total_reward # Total accumulated reward
            }
        
    #--------- REWARDS FUNCTIONS -----------------
    def update_reward(self):
        """
        Recalculates the progress reward and updates the total reward.

        Returns:
            float: The change in total reward from the previous step.
        """
        self.progress_reward = self.calculate_reward() # Calculate current reward components

        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )

        new_total -= self.step_count * self.step_discount # Apply step-based discount
        new_step = new_total - self.total_reward # Calculate the reward gained in this step

        self.total_reward = new_total # Update the total accumulated reward

        return new_step
    
    def calculate_reward(self):
        """
        Calculates and returns a dictionary of various reward components.

        Returns:
            dict: A dictionary where keys are reward component names and values are their calculated rewards.
        """
        return {
            "event": self.reward_scale * self.memory_reader.read_events_done(), # Reward for completed events
            "heal": self.reward_scale * self.total_healing_rew * 0.0005, # Reward for healing
            "dead": self.reward_scale * self.died_count, # Penalty for deaths
            "badge": self.reward_scale * self.batges_in_possesion * 10, # Reward for obtaining badges
            "explore": self.reward_scale * self.get_exploration_reward(), # Reward for exploration (unique cells, regions)
            "region": self.reward_scale * self.get_region_reward(), # Reward for entering new regions
            "coord_explored" : self.reward_scale * len(self.seen_coords)*0.02, # Reward for unique coordinates explored
        }
    
    def update_heal_reward(self):
        """
        Updates the total healing reward and death count based on HP changes.
        Also handles the special case of obtaining the first Pokémon.
        """
        cur_health = self.read_hp_fraction() # Get current health fraction
        
        # Special handling for obtaining the first Pokémon
        if self.last_health == 0 and cur_health > 0 and not self.has_obtained_first_pokemon:
            self.has_obtained_first_pokemon = True # Mark that the first Pokémon has been obtained
            self.last_health = cur_health # Update last_health for the next step
            return

        # If health increased and party size remained the same (implies healing, not new Pokémon)
        if cur_health > self.last_health and self.party_size == self.last_party_size:
            if self.last_health > 0: # Only reward if not starting from 0 HP
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount # Reward based on square of heal amount
            else: # If health goes from 0 to >0, it means a Pokémon revived or was acquired after fainting
                self.died_count += 1 # Count as a death if health was 0 and now is >0 (revive scenario)
        self.last_health = cur_health # Update last health for the next step
    
    def get_exploration_reward(self):
        """
        Calculates the exploration reward based on the current region's exploration progress.
        Rewards are accumulated for each region as it gets explored.

        Returns:
            float: The total accumulated exploration reward.
        """
        region = self.get_current_region()
        # If no region is identified (e.g., initial state or invalid coordinates),
        # return the current sum of regional exploration rewards.
        if not region:
            return sum(self.reward_region_exploration)

        region_reward = 1 - self.get_remaining_in_current_region() # 1 if fully explored, 0 if not at all explored

        if region_reward == 1: # If the current region is fully explored (or met threshold)
           self.reward_region_exploration[int(region["id"])] = 3 # Assign a higher fixed reward
        else:
           self.reward_region_exploration[int(region["id"])] = region_reward # Assign a scaled reward based on progress
        return sum(self.reward_region_exploration) # Return the sum of all regional exploration rewards
    
        
    def update_visit_map(self):
        """
        Increments the visit counter for the current global map cell in `visit_count_map`.
        Only updates if not in combat.
        """
        if not self.is_in_battle:
            gy, gx = self.get_global_coords() # Get global coordinates

            # Ensure coordinates are within map bounds before incrementing
            if 0 <= gy < self.visit_count_map.shape[0] and 0 <= gx < self.visit_count_map.shape[1]:
                self.visit_count_map[gy, gx] += 1

    def get_visit_map_crop(self):
        """
        Returns a cropped and resized portion of the global visit count map,
        centered around the agent's current global coordinates.
        Values in the crop are clipped and then normalized.

        Returns:
            np.array: A 3D numpy array representing the normalized and cropped visit map.
                      Shape will be (output_shape_main[0], output_shape_main[1], 1).
        """
        c = self.get_global_coords() # Get global coordinates (gy, gx)
        crop_initial_size = self.coords_pad * 2 # Initial size of the crop (e.g., 24x24)
        crop = np.zeros((crop_initial_size, crop_initial_size), dtype=np.float32) # Initialize empty crop

        # Check if agent's global coordinates are valid
        if 0 <= c[0] < self.visit_count_map.shape[0] and 0 <= c[1] < self.visit_count_map.shape[1]:
            # Calculate crop boundaries ensuring they stay within the visit_count_map
            y_start = max(0, c[0] - self.coords_pad)
            y_end = min(self.visit_count_map.shape[0], c[0] + self.coords_pad)
            x_start = max(0, c[1] - self.coords_pad)
            x_end = min(self.visit_count_map.shape[1], c[1] + self.coords_pad)

            # Extract the portion of the visit map
            extracted_crop = self.visit_count_map[y_start:y_end, x_start:x_end].astype(np.float32)

            clipped_crop = np.clip(extracted_crop, 0, 1000) # Clip values to a maximum of 1000

            # Calculate where the extracted_crop should be placed within the `crop` array
            # This handles cases where the agent is near the edge of the global map
            out_y_start = self.coords_pad - (c[0] - y_start)
            out_y_end = out_y_start + (y_end - y_start)
            out_x_start = self.coords_pad - (c[1] - x_start)
            out_x_end = out_x_start + (x_end - x_start)

            # Place the clipped extracted crop into the center of the `crop` array
            crop[out_y_start:out_y_end, out_x_start:out_x_end] = clipped_crop

        target_height, target_width = self.output_shape_main[0], self.output_shape_main[1]
        # Resize the crop to the target observation shape using nearest neighbor interpolation
        resized_crop = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # Add a channel dimension to match the observation space shape
        return np.expand_dims(resized_crop, axis=-1)

    
    def get_global_coords(self):
        """
        Converts local (x, y) coordinates and map ID (m) to global (grid) coordinates.

        Returns:
            tuple: A tuple (global_y, global_x) representing the agent's position on the global map.
        """
        return local_to_global(self.y, self.x, self.m)
    
    def segmented_screen(self,screen):
        """
        Processes the raw screen image using a vision model for segmentation.
        If a video recorder is available, it also saves the screen and segmented maps.

        Args:
            screen (np.array): The raw screen image from the emulator.

        Returns:
            np.array: The segmented screen image.
        """
        if self.video_recorder:
            pred, maps = self.vision_model.predict_with_overlay(screen)
            self.video_recorder.save_videos(screen=screen, maps=maps)
        else:
            pred = self.vision_model.predict(screen)
        return pred
    
    def reduce_screen(self,screen):
        """
        Reduces the resolution of a given screen image by half using area interpolation.

        Args:
            screen (np.array): The input screen image.

        Returns:
            np.array: The resized screen image.
        """
        # Resize to half dimensions (width // 2, height // 2) using INTER_AREA for downsampling
        return cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2), interpolation=cv2.INTER_AREA)
    
    
    def get_current_region(self):
        """
        Identifies the current map region based on the agent's map ID (self.m).

        Returns:
            dict or None: A dictionary containing the region's properties if found, otherwise None.
        """
        for r in self.regions:
            if int(r["id"]) == int(self.m): # Compare current map ID with region ID
                return r
        return None # Return None if no matching region is found
    
    def get_region_reward(self):
        """
        Calculates a reward for entering new map regions.
        The reward is accumulated and returned.

        Returns:
            float: The total accumulated reward for visiting new regions.
        """
        region = self.get_current_region()
        # If no region is found (e.g., invalid map ID), return the current accumulated region reward.
        if not region:
            return self.region_count_r 

        region_id = region["id"]

        if region_id not in self.visited_regions: # If this region has not been visited before
            self.visited_regions.add(region_id) # Add it to the set of visited regions
            self.region_count_r += 2 # Increment the region count reward

        return self.region_count_r # Return the total accumulated region reward
    
    def get_remaining_in_current_region(self):
        """
        Calculates the ratio of unexplored cells in the current region.
        This is used to provide a reward based on regional exploration progress.

        Returns:
            float: The remaining unexplored ratio (between 0 and 1).
                   0.0 if the region is explored beyond a threshold, 1.0 if no region or completely unexplored.
        """
        region = self.get_current_region()

        if not region: # If no region is identified
            return 1.0 # Treat as fully unexplored
        
        region_id = region["id"]

        if region_id not in self.region_cells_seen: # Initialize if region not tracked yet
            self.region_cells_seen[region_id] = set()

        self.region_cells_seen[region_id].add((self.x, self.y)) # Add current local coordinates to seen cells

        # Calculate total cells in the region
        w = region["xmax"] - region["xmin"]
        h = region["ymax"] - region["ymin"]
        total_cells = w * h
        visited_cells = len(self.region_cells_seen[region_id])
        
        explored_ratio = visited_cells / total_cells # Calculate the ratio of visited cells

        threshold = 0.75 # Threshold for considering a region "fully" explored
        if explored_ratio >= threshold:
            return 0.0 # Return 0 if explored beyond threshold (meaning 0 remaining)
        else:
            return 1.0 - (explored_ratio / threshold) # Return remaining ratio scaled by threshold