"""
Docstring for Videdit.Layer1.type
What does Videodata look like to a computer?
That's what this file is for.
Were displaying down the metadata, the concepts and phrases that represent a video.
Mapping all these to how we can manipulate them in code and how the computer view it.

"""

from dataclasses import dataclass

class VideoInput:
    path : str
    capture: "cv.VideoCapture"
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int