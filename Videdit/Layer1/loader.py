import cv2 as cv
from .type import VideoInput
import os

class VideoLoadError(Exception):
    pass

def load_video(path: str) -> cv.VideoCapture:
    if not isinstance(path, str):
        raise VideoLoadError("Path must be str")
    if not os.path.exists():
        raise VideoLoadError(f"File not found: {path}")
    
    clip = cv.VideoCapture(path)
    if not clip.isOpened():
        raise VideoLoadError(f"Failed to load video: {path}")
    
    return clip
