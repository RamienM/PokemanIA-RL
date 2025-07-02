import asyncio
import websockets
import json

import gymnasium as gym

# Addresses in RAM for X position, Y position and Map number
X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

class StreamWrapper(gym.Wrapper):
    """
    Adapted from: https://github.com/PWhiddy/PokemonRedExperiments/blob/master/v2/stream_agent_wrapper.py

    Gym environment wrapper that streams game state information (coordinates and map) 
    over a WebSocket connection at a fixed interval.

    It connects to a WebSocket server and periodically sends the accumulated 
    player coordinates and metadata as JSON messages.
    """

    def __init__(self, env, stream_metadata={}):
        """
        Initializes the StreamWrapper.

        :param env: The Gym environment to wrap.
        :param stream_metadata: Additional metadata to include in the streamed messages.
        """
        super().__init__(env)

        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        
        # Create or get an asyncio event loop to manage async websocket communication
        self.loop = asyncio.new_event_loop()

        # Avoid creating multiple event loops if one is already running
        if not asyncio.get_event_loop().is_running():
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.websocket = None

        # Establish WebSocket connection immediately
        self.loop.run_until_complete(self.establish_wc_connection())

        self.upload_interval = 150  # Number of steps between sending messages
        self.steam_step_counter = 0

        self.env = env
        self.coord_list = []  # Accumulates coordinates to send in batch

        # Attempt to find the emulator object from the wrapped environment
        if hasattr(env, "emulator"):
            self.emulator = env.emulator
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def step(self, action):
        """
        Overrides the environment's step method.

        Adds the current player coordinates and map number to the coord_list.

        Sends the accumulated coordinates over the WebSocket every 'upload_interval' steps.

        :param action: The action to take in the environment.
        :return: The usual environment step output (obs, reward, done, info).
        """
        # Read RAM state to get player coordinates and current map
        x_pos = self.emulator.get_ram_state()[X_POS_ADDRESS]
        y_pos = self.emulator.get_ram_state()[Y_POS_ADDRESS]
        map_n = self.emulator.get_ram_state()[MAP_N_ADDRESS]

        self.coord_list.append([x_pos, y_pos, map_n])

        # If upload interval reached, send the batched coordinates
        if self.steam_step_counter >= self.upload_interval:
            # Prepare JSON message including metadata and coordinates
            message = json.dumps({
                "metadata": self.stream_metadata,
                "coords": self.coord_list
            })

            # Send message asynchronously via the event loop
            self.loop.run_until_complete(self.broadcast_ws_message(message))

            # Reset step counter and coordinate list
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        # Step the wrapped environment normally
        return self.env.step(action)
    
    def close(self):
        """
        Overrides environment close method to cleanly close the WebSocket connection and event loop.

        Prevents resource leakage from unclosed WebSocket connections.
        """
        try:
            if self.websocket is not None:
                self.loop.run_until_complete(self.websocket.close())
            if self.loop.is_running():
                self.loop.stop()
            self.loop.close()
        except Exception as e:
            print("Error closing WebSocket:", e)

        # Call the original environment's close method
        super().close()

    async def broadcast_ws_message(self, message):
        """
        Sends a message over the WebSocket connection asynchronously.

        Re-establishes the connection if it was lost.

        :param message: JSON string to send.
        """
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                # On error, reset the websocket to None to try reconnecting later
                self.websocket = None

    async def establish_wc_connection(self):
        """
        Establishes a WebSocket connection asynchronously.

        Sets websocket to None if connection fails.
        """
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except Exception:
            self.websocket = None
