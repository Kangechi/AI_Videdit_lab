import cv2 as cv

def extract_metadata(clip: cv.VideoCapture) -> dict:
    fps = clip.get(cv.CAP_PROP_FPS)
    frame_count = int(clip.get(cv.CAP_PROP_FRAME_COUNT))
    width =int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or frame_count <= 0:
        raise ValueError("Invalid video metadata of frame count and fps")
    
    duration = frame_count / fps

    return {
        "fps":fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height

    }
  