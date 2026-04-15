"""OpenCV-based video sources with simple multi-camera simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VideoFrame:
    camera_id: str
    frame_index: int
    frame_bgr: object


class OpenCVSource:
    def __init__(self, camera_id: str, source: int | str, loop_video: bool = True) -> None:
        self.camera_id = camera_id
        self.source = source
        self.loop_video = loop_video
        self.capture = cv2.VideoCapture(source)
        
        # Verify camera opened successfully
        if not self.capture.isOpened():
            logger.error(f"Failed to open camera '{camera_id}' with source: {source}")
        else:
            logger.info(f"Successfully opened camera '{camera_id}' with source: {source}")

    def read(self, frame_index: int) -> VideoFrame | None:
        if not self.capture.isOpened():
            logger.warning(f"Camera '{self.camera_id}' is not opened")
            return None

        ok, frame = self.capture.read()
        if not ok and isinstance(self.source, str) and self.loop_video:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()

        if not ok:
            logger.warning(f"Failed to read frame from camera '{self.camera_id}'")
            return None
        return VideoFrame(camera_id=self.camera_id, frame_index=frame_index, frame_bgr=frame)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            logger.info(f"Released camera '{self.camera_id}'")


def detect_available_cameras(max_index: int = 30) -> list[int]:
    """Detect available camera indices on the system.
    
    Args:
        max_index: Maximum camera index to check (0 to max_index-1)
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    for camera_index in range(max_index):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            available_cameras.append(camera_index)
            cap.release()
            logger.info(f"Found camera at index {camera_index}")
        else:
            cap.release()
    
    if not available_cameras:
        logger.warning("No cameras detected on this system!")
    
    return available_cameras


def build_sources(mode: str, video_path: str | None, camera_count: int) -> list[OpenCVSource]:
    mode = mode.lower()
    if mode == "webcam":
        # Detect available cameras instead of hardcoding to 0
        available_cameras = detect_available_cameras()
        if not available_cameras:
            logger.error("No cameras found. Please check camera connection and permissions.")
            return []
        
        # Use first available camera
        camera_source = available_cameras[0]
        logger.info(f"Using camera index {camera_source}")
        return [OpenCVSource(camera_id="cam-1", source=camera_source)]

    if mode == "video":
        if not video_path:
            logger.error("Video path is required for 'video' mode")
            return []
        return [OpenCVSource(camera_id="cam-1", source=video_path)]

    if mode == "multi":
        if not video_path:
            logger.error("Video path is required for 'multi' mode")
            return []
        return [OpenCVSource(camera_id=f"cam-{idx + 1}", source=video_path) for idx in range(camera_count)]

    logger.error(f"Unknown video source mode: {mode}")
    return []
