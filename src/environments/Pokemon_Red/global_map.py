# Adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py

import os
import json

# Path to the JSON file containing map region data
MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")

# Padding added around the global map to avoid boundary issues
PAD = 20

# Global map dimensions including padding (rows, columns)
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)

# Offsets to translate local map coordinates into global map indices
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

# Load region data from JSON and index by region ID as integer keys
with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}


def local_to_global(r: int, c: int, map_n: int):
    """
    Converts local coordinates (r, c) on a specific map (map_n)
    to global coordinates in the overall game world map.
    
    Parameters:
    - r (int): local row coordinate within the map
    - c (int): local column coordinate within the map
    - map_n (int): ID of the map region
    
    Returns:
    - Tuple (gy, gx): global coordinates within GLOBAL_MAP_SHAPE
    
    If the map ID is not found or coordinates are out of bounds,
    returns the global map center as a fallback.
    """
    try:
        # Get the global (x, y) offset of the map region
        map_x, map_y = MAP_DATA[map_n]["coordinates"]

        # Compute global coordinates by adding offsets and padding
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET

        # Check if the computed coordinates are within global map bounds
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        
        # Warn if out-of-bounds and return center fallback
        print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2

    except KeyError:
        # Warn if map ID not found and return center fallback
        print(f"Map id {map_n} not found in map_data.json.")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
