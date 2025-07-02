import cv2
import os
import numpy as np

class VideoRecorder:
    def __init__(self, frame_width=160, frame_height=144, fps=30, flush_every=10000, save_path="game&results/Pokemon_red_env/videos"):
        """
        Initializes the VideoRecorder.

        :param frame_width: Width of the video frames.
        :param frame_height: Height of the video frames.
        :param fps: Frames per second for the video.
        :param flush_every: Number of frames after which video writers are restarted to save files in parts.
        :param save_path: Directory path to save the video files.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.flush_every = flush_every
        self.save_path = save_path

        self.screen_video_writer = None
        self.maps_video_writer = None

        self.frame_count = 0
        self.part_number = 0  # To save videos in numbered parts

        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _start_new_writers(self):
        """
        Initializes new video writers for screen and maps videos, 
        using incremented part numbers in filenames.
        """
        screen_path = os.path.join(self.save_path, f"screen_video_{self.part_number:03}.mp4")
        maps_path = os.path.join(self.save_path, f"maps_video_{self.part_number:03}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.screen_video_writer = cv2.VideoWriter(screen_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.maps_video_writer = cv2.VideoWriter(maps_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def start_writing(self):
        """
        Starts the video writers by initializing them.
        """
        self._start_new_writers()

    def _release_writers(self):
        """
        Releases and closes the video writers safely.
        """
        if self.screen_video_writer:
            self.screen_video_writer.release()
            self.screen_video_writer = None
        if self.maps_video_writer:
            self.maps_video_writer.release()
            self.maps_video_writer = None

    def save_videos(self, screen, maps):
        """
        Saves a pair of frames (screen and maps) to the respective video files.

        :param screen: Frame from the game screen (expected as numpy array).
        :param maps: Frame from the maps (expected as numpy array, float [0,1]).
        :raises RuntimeError: If the video writers are not initialized.
        """
        if self.screen_video_writer is None or self.maps_video_writer is None:
            raise RuntimeError("Videos are not initialized. Call start_writing() first.")

        # Convert screen to BGR if grayscale (single channel)
        if len(screen.shape) == 3 and screen.shape[2] == 1:
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
        elif len(screen.shape) == 2:
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)

        # Convert maps from float [0,1] to uint8 [0,255] and from RGB to BGR
        maps_uint8 = (maps * 255).astype(np.uint8)
        maps_uint8 = cv2.cvtColor(maps_uint8, cv2.COLOR_RGB2BGR)

        # Write frames to the video files
        self.screen_video_writer.write(screen)
        self.maps_video_writer.write(maps_uint8)

        self.frame_count += 1

        # After flush_every frames, close current videos and start new ones to avoid large file sizes
        if self.frame_count >= self.flush_every:
            self._release_writers()
            self.part_number += 1
            self._start_new_writers()
            self.frame_count = 0

    def stop_writing(self):
        """
        Stops video recording by releasing video writers.
        """
        self._release_writers()
