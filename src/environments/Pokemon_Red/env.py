from src.environments.Pokemon_Red.global_map import local_to_global, GLOBAL_MAP_SHAPE

import uuid
from gymnasium import Env, spaces
from collections import deque
import numpy as np
import json 
import cv2

class PokemonRedEnv(Env):

    def __init__(self, emulator, memory_reader, vision_model, video_recorder, config):
        self.emulator = emulator
        self.memory_reader = memory_reader
        self.vision_model = vision_model
        self.video_recorder = video_recorder
        

        ##--------- Config -----------------
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        
        self.enc_freqs = (
            5 if "fourier_encode_freq" not in config else config["fourier_encode_freq"]
        )
        self.quantity_action_storage = (
            5 if "actions_stack" not in config else config["actions_stack"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.max_steps = (
            1000 if "max_steps" not in config else config["max_steps"]
        )
        self.init_state = (
            "../states/STELLE.state" if "init_state" not in config else config["init_state"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.coords_pad = (
            12 if "coords_pad" not in config else config["coords_pad"]
        )

        self.step_discount = config.get("step_discount", 0)

        #--------- Observation space config -----------------
        self.recent_actions = deque(maxlen=self.quantity_action_storage)
        self.action_space = spaces.Discrete(len(self.emulator.VALID_ACTIONS))

        self.output_shape_main = (72,80, self.quantity_action_storage)
        self.observation_space = spaces.Dict(
            {
                "main_screen": spaces.Box(low=0, high=255, shape=self.output_shape_main, dtype=np.uint8),
                "segmented_screen" : spaces.Box(low=0, high=15, shape=self.output_shape_main, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "badges": spaces.Discrete(8),
                "events": spaces.MultiBinary(self.memory_reader.get_difference_between_events()),
                #"map": spaces.Box(low=0, high=255, shape=(self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "visit_map": spaces.Box(low=0.0, high=1.0, shape=(self.coords_pad*4, self.coords_pad*4, 1), dtype=np.float32),
                "recent_actions": spaces.MultiDiscrete([len(self.emulator.VALID_ACTIONS)]*self.quantity_action_storage),
                "remaining_ratio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                #"coords": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.int32),
                "in_combat" : spaces.MultiBinary(1),
            }
        )

        self.reset_count = 0

        #--------- Video recorder -----------------
        if self.video_recorder:
            self.video_recorder.start_writing()

        #---------Loading regions -----------------
        regions_path = "src/environments/Pokemon_Red/map_data.json"
        with open(regions_path, "r") as f:
            data = json.load(f)
        self.regions = []
        for r in data["regions"]:
            x0, y0 = r["coordinates"]
            w, h = r["tileSize"]
            # cada región: [x_min, x_max, y_min, y_max, id, name]
            self.regions.append({
                "id": r["id"],
                "name": r["name"],
                "xmin": x0,
                "xmax": x0 + w,
                "ymin": y0,
                "ymax": y0 + h,
            })
        
        #---------Loading events -----------------
        with open("src/environments/Pokemon_Red/events.json") as f:
            event_names = json.load(f)
        self.event_descriptions_by_addr_bit  = event_names


    def reset(self, seed=None, options=None):
        self.seed = seed
        self.emulator.load_state(self.init_state)

        self.seen_coords = {}

        self.visit_count_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)

        self.recent_main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        self.recent_seg_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        
        self.recent_actions = deque([0] * self.quantity_action_storage, maxlen=self.quantity_action_storage)


        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.has_obtained_first_pokemon = False 
        self.party_size = 0
        self.last_party_size = 0
        self.step_count = 0
        self.region_count_r = 0
        self.stuck = 0

        self.is_in_battle = False
        self.batges_in_possesion = 0
        self.x, self.y, self.m = 0,0,0
        ##Vision

        #Events
        self.base_event_flags = self.memory_reader.read_events_done()
        self.all_event_flags_set_history = set()
        self.current_new_events = []
        self.last_recorded_event_description = []

        self.visited_regions = set()
        self.region_cells_seen = {}  # id_region -> set de (x, y)

        for region in self.regions:
            self.region_cells_seen[region["id"]] = set()
        
        self.reward_region_exploration = [0]*len(self.regions)

        self.progress_reward = self.calculate_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}
        
    def step(self, action):

        self.emulator.step(action)
        self.update_recent_actions(action)

        self.x,self.y,self.m = self.memory_reader.get_game_coords()
        self.is_in_battle = self.memory_reader.is_in_battle()
        self.batges_in_possesion = self.memory_reader.read_bagdes_in_possesion()
        self.party_size = self.memory_reader.read_pokemon_in_party()

        self.update_seen_coords()
        self.update_heal_reward()
        self.update_visit_map()
        self.update_events()
        #self.compute_global_stuck()


        new_reward = self.update_reward()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()
        self.print_info()

        self.last_party_size = self.party_size 
        self.step_count += 1
        info = {"episode": {"r": self.total_reward,  "l": self.step_count, "exploration_reward":self.reward_scale * (self.get_exploration_reward()) * 3}}
        return obs, new_reward, False, step_limit_reached, info

    def _get_obs(self):
        screen = self.emulator.get_screen()
        segmentation = self.segmented_screen(screen)

        reduced_screen = self.reduce_screen(screen[:,:,0])
        reduced_segmentation = self.reduce_screen(segmentation)

        self.update_recent_stack(self.recent_main_screen, reduced_screen)
        self.update_recent_stack(self.recent_seg_screen, reduced_segmentation)

        observation = {
            "main_screen": self.recent_main_screen, 
            "segmented_screen": self.recent_seg_screen,
            "health": np.array([self.read_hp_fraction()]),
            "badges": self.batges_in_possesion,
            "events": np.array(self.memory_reader.read_event_bits(), dtype=np.int8),
            #"map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "remaining_ratio":  np.array([self.get_remaining_in_current_region()], dtype=np.float32),
            #"coords": np.array([self.x,self.y,self.m], dtype=np.int32),
            "visit_map": self.get_visit_map_crop()[..., None],  # visitas normalizadas
            "in_combat": np.array([int(self.is_in_battle)], dtype=np.uint8)
        }
        
        return observation
    
    def check_if_done(self):
        return self.step_count >= self.max_steps - 1

    def print_info(self):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.5f}"
            prog_string += f" sum: {self.total_reward:5.5f}"
            print(f"\r{prog_string}", end="", flush=True)
    
    def update_recent_actions(self, action):
        self.recent_actions.appendleft(action)
    
    def update_events(self):
        current_active_addr_bit_flags = set()
        
        raw_event_bits = self.memory_reader.read_event_bits()
        event_flags_start_addr, event_flags_end_addr = self.memory_reader.get_event_flags()

        total_addresses = event_flags_end_addr - event_flags_start_addr
        expected_total_bits = total_addresses * 8

        if len(raw_event_bits) != expected_total_bits:
            print(f"DEBUG: Desajuste en la longitud de bits de eventos esperada. Se esperaban {expected_total_bits}, se obtuvieron {len(raw_event_bits)}. Procesando datos parciales.")

        bit_counter = 0
        for address in range(event_flags_start_addr, event_flags_end_addr):
            for bit_idx_in_byte in range(8):
                if bit_counter < len(raw_event_bits):
                    if raw_event_bits[bit_counter] == 1:
                        key = f"0x{address:X}-{bit_idx_in_byte}"
                        current_active_addr_bit_flags.add(key)
                    bit_counter += 1
                else:
                    break 
            if bit_counter >= len(raw_event_bits):
                break 

        newly_activated_flags = current_active_addr_bit_flags - self.all_event_flags_set_history
        
        self.current_new_events = [] 
        for flag_key in newly_activated_flags:
            if flag_key in self.event_descriptions_by_addr_bit:
                self.current_new_events.append(self.event_descriptions_by_addr_bit[flag_key])
            else:
                self.current_new_events.append(f"Unknown Event: {flag_key}")
                print(f"DEBUG: Clave de evento desconocida en newly_activated_flags: {flag_key}")
        
        self.all_event_flags_set_history.update(current_active_addr_bit_flags)

        if self.current_new_events: # Si se encontraron nuevos eventos en este paso
            self.last_recorded_event_description = list(self.current_new_events)
    
    def update_recent_stack(self, stack, new_frame):
        stack[:, :, 1:] = stack[:, :, :-1]  # desplaza a la derecha
        stack[:, :, 0] = new_frame
    
    def update_seen_coords(self):
        """
        Registra las coordenadas visitadas, contando las visitas y guardando el paso en que se visitaron.
        
        Parámetros:
        - current_step: Número de paso actual en la simulación.
        """
        if not self.is_in_battle:
            coord_string = f"x:{self.x} y:{self.y} m:{self.m}"
            

            # Incrementar el número de veces que se ha visitado la coordenada
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string]['count'] += 1
            else:
                self.seen_coords[coord_string] = {'count': 1}
            
    
    def read_hp_fraction(self):
        hp_sum =  self.memory_reader.get_sum_all_current_hp()
        max_hp_sum = self.memory_reader.get_sum_all_max_hp()

        if max_hp_sum == 0:
            return hp_sum
        
        return hp_sum / max_hp_sum
    
    def get_agent_stats(self):
        levels = self.memory_reader.get_all_player_pokemon_level()
        reg = self.get_current_region()
        region_id   = reg["id"]   if reg else None
        region_name = reg["name"] if reg else "Unknown"
        region_reward = self.get_region_reward()

        return {
                "step": self.step_count,
                "x": self.x,
                "y": self.y,
                "region_id": region_id,
                "region_name": region_name,
                "region_reward": region_reward,
                "pcount": self.party_size,
                "team_p": self.memory_reader.get_all_player_pokemon_name(),
                "levels": levels,
                "levels_sum": sum(levels),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "explore": self.get_exploration_reward(),
                "deaths": self.died_count,
                "badge": self.batges_in_possesion,
                "event": self.progress_reward["event"],
                "last_event_done": self.last_recorded_event_description,
                "healr": self.total_healing_rew,
                "action": self.recent_actions[0],
                "in_combat": self.is_in_battle,
                "total_reward" : self.total_reward
            }
        
    #--------- REWARDS FUNCTIONS -----------------
    def update_reward(self):
        self.progress_reward = self.calculate_reward()

        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )

        if not self.is_in_battle:
            new_total -= self.step_count * self.step_discount 
        else:
            new_total -= self.step_count * self.step_discount / 5
        new_step = new_total - self.total_reward

        self.total_reward = new_total

        return new_step
    
    def calculate_reward(self):
        return {
            "event": self.reward_scale * self.memory_reader.read_events_done(),
            "heal": self.reward_scale * self.total_healing_rew * 0.0003,
            "dead": self.reward_scale * self.died_count,
            "badge": self.reward_scale * self.batges_in_possesion * 10,
            "explore": self.reward_scale * self.get_exploration_reward() * 2,
            "region": self.reward_scale * self.get_region_reward(),
            "coord_explored" : self.reward_scale * len(self.seen_coords)*0.001
        }
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if self.last_health == 0 and cur_health > 0 and not self.has_obtained_first_pokemon:
            self.has_obtained_first_pokemon = True # Marca que ya se obtuvo el primer Pokémon
            self.last_health = cur_health # Actualiza last_health para el siguiente paso
            return

        if cur_health > self.last_health and self.party_size == self.last_party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1
        self.last_health = cur_health
    
    def get_exploration_reward(self):
        region = self.get_current_region()
        region_reward = 1-self.get_remaining_in_current_region()

        if region_reward == 1:
           self.reward_region_exploration[int(region["id"])] = 3
        else:
           self.reward_region_exploration[int(region["id"])] = region_reward
        return sum(self.reward_region_exploration)
    
    def compute_global_stuck(self):
        """
        Calcula un valor de 'stuck' basado en el número de casillas visitadas más de una vez.
        """
        self.stuck = sum( 1 for data in self.seen_coords.values() if data['count'] > 1)
        return self.stuck
    
    def get_explore_map(self):
        """
        Devuelve el recorte del mapa de exploración (visitado/no visitado)
        derivado de visit_count_map. Los valores serán 0 (no visitado) o 255 (visitado).
        """
        c = self.get_global_coords()
        
        # Inicializamos 'out' con ceros para casos fuera de límites o si el mapa está vacío
        # El tamaño es coords_pad*2 x coords_pad*2, y el tipo es uint8 (para la salida 0 o 255)
        out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)

        # Verificamos si las coordenadas están dentro de los límites del visit_count_map
        if 0 <= c[0] < self.visit_count_map.shape[0] and 0 <= c[1] < self.visit_count_map.shape[1]:
            # Extraemos el recorte del visit_count_map
            # Usamos un slice seguro para evitar errores si el centro está cerca del borde
            # y el recorte se extendería fuera del mapa.
            # Aquí, 'slice' garantiza que solo se accedan a los índices válidos.
            y_start = max(0, c[0] - self.coords_pad)
            y_end = min(self.visit_count_map.shape[0], c[0] + self.coords_pad)
            x_start = max(0, c[1] - self.coords_pad)
            x_end = min(self.visit_count_map.shape[1], c[1] + self.coords_pad)

            temp_crop_from_visits = self.visit_count_map[y_start:y_end, x_start:x_end]
            
            # Convertimos a binario: 255 si la cuenta es > 0 (visitado), 0 si es 0 (no visitado)
            # Aseguramos que el tamaño del 'out' final sea el deseado, rellenando con ceros si el recorte es más pequeño
            
            # Calculamos dónde debería ir el recorte extraído dentro del 'out' final
            out_y_start = self.coords_pad - (c[0] - y_start)
            out_y_end = out_y_start + (y_end - y_start)
            out_x_start = self.coords_pad - (c[1] - x_start)
            out_x_end = out_x_start + (x_end - x_start)

            out[out_y_start:out_y_end, out_x_start:out_x_end] = (temp_crop_from_visits > 0).astype(np.uint8) * 255
    
        # Redimensionamos el recorte final (out)
        out = cv2.resize(out, (out.shape[1]*2, out.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        return out
        
    def update_visit_map(self):
        """
        Aumenta en 1 el contador de visitas de la casilla actual en visit_count_map.
        """
        if not self.is_in_battle:
            gy, gx = self.get_global_coords()

            if 0 <= gy < self.visit_count_map.shape[0] and 0 <= gx < self.visit_count_map.shape[1]:
                self.visit_count_map[gy, gx] += 1

    def get_visit_map_crop(self):
        """
        Devuelve el recorte del mapa de visitas, con valores limitados a 1000.
        """
        c = self.get_global_coords()
        crop = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.float32)

        if 0 <= c[0] < self.visit_count_map.shape[0] and 0 <= c[1] < self.visit_count_map.shape[1]:
            # Usamos slice seguro también aquí, como en get_explore_map
            y_start = max(0, c[0] - self.coords_pad)
            y_end = min(self.visit_count_map.shape[0], c[0] + self.coords_pad)
            x_start = max(0, c[1] - self.coords_pad)
            x_end = min(self.visit_count_map.shape[1], c[1] + self.coords_pad)

            extracted_crop = self.visit_count_map[y_start:y_end, x_start:x_end].astype(np.float32)

            # Aplicamos el límite de 1000
            clipped_crop = np.clip(extracted_crop, 0, 1000)

            # Rellenamos el 'crop' final con el recorte obtenido, asegurando el tamaño correcto
            out_y_start = self.coords_pad - (c[0] - y_start)
            out_y_end = out_y_start + (y_end - y_start)
            out_x_start = self.coords_pad - (c[1] - x_start)
            out_x_end = out_x_start + (x_end - x_start)

            crop[out_y_start:out_y_end, out_x_start:out_x_end] = clipped_crop

        # Ampliamos
        crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        return crop
    
    def get_global_coords(self):
        return local_to_global(self.y, self.x, self.m)
    
    def segmented_screen(self,screen):
        if self.video_recorder:
            pred, maps = self.vision_model.predict_with_overlay(screen)
            self.video_recorder.save_videos(screen=screen,maps=maps)
        else:
            pred = self.vision_model.predict(screen)
        return pred
    
    def reduce_screen(self,screen):
        return cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2), interpolation=cv2.INTER_AREA)
    
    
    def get_current_region(self):
        """
        Dado x,y absolutos, devuelve la región que los contiene,
        o None si no está en ninguna.
        """
        for r in self.regions:
            if int(r["id"]) == int(self.m):
                return r
        return None
    
    def get_region_reward(self):
        """
        Devuelve una recompensa si el agente entra en una nueva región.
        """
        region = self.get_current_region()
        if not region:
            return self.region_count_r 

        region_id = region["id"]

        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            self.region_count_r += 5

        return self.region_count_r
    
    def get_remaining_in_current_region(self):
        region = self.get_current_region()

        
        if not region:
            return 1.0  # No estás en ninguna región, asumimos sin explorar

        region_id = region["id"]

        # Inicializa el set si aún no existe
        if region_id not in self.region_cells_seen:
            self.region_cells_seen[region_id] = set()

        # Marca la celda como visitada
        self.region_cells_seen[region_id].add((self.x, self.y))

        # Calcula progreso
        w = region["xmax"] - region["xmin"]
        h = region["ymax"] - region["ymin"]
        total_cells = w * h
        visited_cells = len(self.region_cells_seen[region_id])
        
        # Cálculo del porcentaje explorado
        explored_ratio = visited_cells / total_cells

        # Queremos que: 
        #   - explored_ratio >= 0.75  -> return 0.0
        #   - explored_ratio == 0.0   -> return 1.0
        #   - explored_ratio in (0, 0.75) -> lineal de 1.0 a 0.0

        threshold = 0.75
        if explored_ratio >= threshold:
            return 0.0
        else:
            return 1.0 - (explored_ratio / threshold)

