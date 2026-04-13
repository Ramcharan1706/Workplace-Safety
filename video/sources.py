"""OpenCV-based video sources with simple multi-camera simulation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2


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

    def read(self, frame_index: int) -> VideoFrame | None:
        if not self.capture.isOpened():
            return None

        ok, frame = self.capture.read()
        if not ok and isinstance(self.source, str) and self.loop_video:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()

        if not ok:
            return None
        return VideoFrame(camera_id=self.camera_id, frame_index=frame_index, frame_bgr=frame)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def build_sources(mode: str, video_path: str | None, camera_count: int) -> list[OpenCVSource]:
    mode = mode.lower()
    if mode == "webcam":
        return [OpenCVSource(camera_id="cam-1", source=0)]

    if mode == "video":
        if not video_path:
            return []
        return [OpenCVSource(camera_id="cam-1", source=video_path)]

    if mode == "multi":
        if not video_path:
            return []
        return [OpenCVSource(camera_id=f"cam-{idx + 1}", source=video_path) for idx in range(camera_count)]

    return []
