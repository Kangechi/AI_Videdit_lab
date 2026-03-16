import os
import cv2 as cv
import numpy as np

"""
Lookig at systems's thinking in writing this code:
1. The data shape for each stage: the input it takes - the shape of that input
2. The output shape the function gives.
3. The guarantess 
4. The failure modes

First function is video reading as well as metadata extraction and validation
 The two validation segments highlight:
 1. Checks whether the input is an accepted shape for the function
 2. The other validates the shape location.

 The flow is the validation of the data shape - of the input
 Then extraction of the metadata, then validation of that metadata then lastly the return value of the output data shape
 This is the stage to fail loudly to prevent any further areas at the end.

 For the function on readin/extracting frames
 It also takes in a string, then gives out the data shape of a list.
 The functionality is to loop over the frames of the video, thus while loop included in the decision making,
 If/ else conditions are also used in breaking the loops and validation

"""
def validate_video(video_path: str) -> dict:
    if not isinstance(video_path, str):
        raise ValueError("Video_path must be string")
    if not os.path.exists(video_path):
        raise ValueError("File path does not exist")
    
    clip = cv.VideoCapture(video_path)
    if not clip.isOpened():
        raise ValueError("Video file unable to open")
    
    fps = clip.get(cv.CAP_PROP_FPS)
    frame_count = int(clip.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0  or frame_count <= 0 :
        raise ValueError("Invalid video metadata")
    clip.release()

    return{
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
      }

def extract_frames(video_path: str) -> list:
    clip = cv.VideoCapture(video_path)
    if not clip.isOpened():
        raise ValueError("Error in oppening video")
    
    frames = []
    while True:
        ret, frame = clip.read()
        frames.append(frame)
        if not ret:
            break

    clip.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted")
    
    return frames

def segmentation_slicing(frames: list, fps: float, segment_seconds: float= 2.0) -> list:
    segment_length =int( fps * segment_seconds)
    segments =[]

    for start in range(0,len(frames), segment_length):
        end = min(start + segment_length, len(frames))
        segments.append(start, end)

    return segments

def cut_detection(frames: list, threshold: float= 15.0) -> list:
    cuts = []
    for i in range(1, len(frames)):
        diff = np.mean(cv.absdiff(frames{i}, frames{i-1}))
        if diff > threshold:
            cuts.append(i)
    
    return cuts

def template_abstarction(metadata: dict, cuts: list) -> dict:
    required_keys = ["fps", "frame_count", "width", "height"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing field {key}")
        
    return{
        "video": {
            "fps":metadata["fps"],
            "resolution": (metadata["width"], metadata["height"]),
            "total_frames":metadata["frame_count"]
        },
        "cuts": cuts,
        "style": {
            "default_transition": "cut"
        }
    }

