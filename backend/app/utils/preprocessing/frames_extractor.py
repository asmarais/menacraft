import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
import logging

class FramesExtractor:
    """Extract frames from video files with configurable sampling rate, extracted frame data 
    and metadata (timestamp, frame number).
    """

    def __init__(self,
        video_path: str,
        nb_fps: Optional[int] =None,
    ):
        """Initialize VideoExtractor
        Parameters:
            video_path (str): path to the video file
            nb_fps (int, optional): number of frames to extract per second.
        Raises:
            FileNotFoundError: if video file doesn't exist
            ValueError: if video fail to open
        """
        self.video_path: Path = Path(video_path)
        """Video file path"""
        self.nb_fps: Optional[int] = nb_fps
        """Number of frames to extract per second"""
        self.logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for the FramesExtractor class"""

        if not self.video_path.exists():
            self.logger.fatal(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._video_props: cv2.VideoCapture = cv2.VideoCapture(str(self.video_path))
        """Captures video properties"""

        if not self._video_props.isOpened():
            self.logger.fatal(f"Could not open video file: {self.video_path}")
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.video_fps = self._video_props.get(cv2.CAP_PROP_FPS)
        """Fps of the video"""
        try:
            assert self.video_fps > 0, "Video has 0 FPS"
            
        except AssertionError as e:
            self._video_props.release()
            self.logger.fatal(f"Invalid parameters: {self.video_fps} for video {self.video_path}")
            raise ValueError(f"Invalid parameters: {self.video_fps} for video {self.video_path}")

        self.total_frames = int(self._video_props.get(cv2.CAP_PROP_FRAME_COUNT))
        """Total number of frames in the video"""
        try:
            assert self.total_frames > 0, "Video has 0 frames"
            
        except AssertionError as e:
            self._video_props.release()
            self.logger.fatal(f"Invalid parameters: {self.total_frames} for video {self.video_path}")
            raise ValueError(f"Invalid parameters: {self.total_frames} for video {self.video_path}")    
    
        self.width = int(self._video_props.get(cv2.CAP_PROP_FRAME_WIDTH))
        """Width of the video frames"""
        self.height = int(self._video_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
        """Height of the video frames"""
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        """Duration of the video in seconds"""
        
        if self.nb_fps is not None and self.nb_fps > 0:
            self.frame_interval = max(1, int(self.video_fps / self.nb_fps))
        else:
            self.frame_interval = 1  #every frame extracted by default
        
    def extract_frames(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames from the video.
        Yields:
            Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]: frames and associated metadata
            - np.ndarray of shape (height, width, 3), frames in BGR format
            - Dict[str, Any] with keys 'frame_id', 'timestamp', 'frame_number'
        Raises:
            RuntimeError: if frame reading fails
            IndexError: if frame_index is out of range
        """
        frame_count = 0
        extracted_count = 0

        try:
            #reset video to beginning
            self._video_props.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while frame_count < self.total_frames:
                #we only extract every n frame according to frame_interval
                if frame_count % self.frame_interval == 0:
                    extracted = self.get_frame_at(frame_count)

                    if extracted is None:
                        self.logger.info(f"- extract_frames - reached end of video at frame {frame_count}.")
                        break

                    yield extracted

                    extracted_count += 1

                frame_count += 1

            self.logger.info(f"- extract_frames - frame extraction complete: {extracted_count} frames extracted from {frame_count} total frames.")

        except Exception as e:
            self.logger.error(f"- extract_frames - {e}")
            raise RuntimeError(f"Error during frame extraction: {e}")
        
    def get_frame_at(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract a specific frame by frame number.
        Parameters:
            frame_index (int): frame index to extract
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (frame, metadata) or None if frame doesn't exist
        Raises:
            IndexError: if frame_index is out of range
        """
        try:
            assert frame_index >= 0 and frame_index < self.total_frames, f"Frame index {frame_index} out of range [0, {self.total_frames})"
            
        except AssertionError as e:
            self.logger.error(f"- get_frame_at - {e}")
            raise IndexError(f"Error getting frame at index {frame_index}: {e}")

        self._video_props.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        is_read, frame = self._video_props.read()

        if not is_read:
            self.logger.debug(f"- get_frame_at - failed to read frame {frame_index}")
            return None

        timestamp = frame_index / self.video_fps if self.video_fps > 0 else 0

        metadata = {
            "frame_id": frame_index,
            "frame_number": frame_index,
            "timestamp": timestamp,
            "video_fps": self.video_fps,
            "resolution": (self.width, self.height)
        }

        return frame, metadata

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information.
        Returns:
            Dict[str, Any]: dictionary with video properties
        """
        return {
            "path": str(self.video_path),
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "frame_interval": self.frame_interval
        }

    def save_frame(self, frame: np.ndarray, output_path: str) -> None:
        """Save a frame to the provided location.
        Parameters:
            frame (np.ndarray): frame to save (numpy array)
            output_path (str): path where to save the frame
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if saving fails
        """
        try:
            assert frame is not None and frame.size > 0, "Invalid frame to save"
            assert Path.exists(Path(output_path).parent), "Output directory does not exist"
        
        except AssertionError as e:
            self.logger.error(f"- save_frame - {e}")
            raise ValueError(f"Invalid parameters for save_frame: {e}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), frame)

        if not success:
            raise RuntimeError(f"Failed to save frame to {output_path}")

    def __del__(self):
        """Destructor - ensure resources are released."""
        if self._video_props is not None:
            self._video_props.release()
