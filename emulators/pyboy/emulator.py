from pyboy import PyBoy 
from pyboy.utils import WindowEvent
import numpy as np

class GameEmulator:
    """
    Wrapper for the PyBoy Game Boy emulator to interact with a game environment.
    Allows programmatic control of inputs, loading of game states, and retrieval
    of screen and memory data.

    Attributes
    ----------
    VALID_ACTIONS : dict
        Maps integer actions to PyBoy input events.
    REALISE_ACTIONS : dict
        Maps press events to their corresponding release events.
    """

    VALID_ACTIONS = {
        0: WindowEvent.PRESS_ARROW_DOWN,
        1: WindowEvent.PRESS_ARROW_LEFT,
        2: WindowEvent.PRESS_ARROW_RIGHT,
        3: WindowEvent.PRESS_ARROW_UP,
        4: WindowEvent.PRESS_BUTTON_A,
        5: WindowEvent.PRESS_BUTTON_B,
        6: WindowEvent.PRESS_BUTTON_START
    }

    REALISE_ACTIONS = {
        WindowEvent.PRESS_ARROW_DOWN : WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT : WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT : WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP : WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A : WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B : WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START : WindowEvent.RELEASE_BUTTON_START
    }
    
    def __init__(self, config):
        """
        Initializes the game emulator.

        Parameters
        ----------
        config : dict
            Dictionary containing configuration settings. Must include "gb_path".
            If "headless" is True, disables the display window.
            Optionally accepts "emulation_speed".
        """
        rom_path = config["gb_path"]
        window_type = "null" if config["headless"] else "SDL2"
        emulation_speed = (
            5 if "emulation_speed" not in config else config["emulation_speed"]
        )

        self.pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
        self.pyboy.set_emulation_speed(emulation_speed)
        
    def step(self, action):
        """
        Performs one step in the environment by simulating a button press and release.

        Parameters
        ----------
        action : int
            Integer corresponding to a valid action in VALID_ACTIONS.

        Raises
        ------
        ValueError
            If the action is not in VALID_ACTIONS.
        """
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be one of {list(self.VALID_ACTIONS.keys())}")
        
        button = self.VALID_ACTIONS[action]
        press_step = 8  # Duration of the press, based on academic paper recommendations

        self.pyboy.send_input(button, True)
        self.pyboy.tick(press_step)
        self.pyboy.send_input(self.REALISE_ACTIONS[button], False)
        self.pyboy.tick(press_step)
        self.pyboy.tick(1)
    
    def load_state(self, initial_state):
        """
        Loads a saved emulator state.

        Parameters
        ----------
        initial_state : str
            Path to the saved state file.
        """
        with open(initial_state, "rb") as f:
            self.pyboy.load_state(f)

    def get_ram_state(self):
        """
        Returns the current state of the emulator's RAM.

        Returns
        -------
        pyboy.memory.Memory
            The memory object giving access to emulator RAM.
        """
        return self.pyboy.memory
    
    def get_screen(self):
        """
        Retrieves the current screen image as a grayscale NumPy array.

        Returns
        -------
        np.ndarray
            Grayscale screen frame with shape (144, 160, 1).
        """
        return np.expand_dims(self.pyboy.screen.ndarray[:, :, 0], axis=-1)
    
    def close(self):
        """
        Properly shuts down the emulator.
        """
        self.pyboy.stop()
