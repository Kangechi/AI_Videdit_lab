from .type import VideoInput
from .loader import load_video
from .metadata import extract_metadata
from .validator import validate_metadata

def ingest_video(path: str) ->VideoInput:
    clip = load_video(r"C:\Users\ADMIN\Desktop\AI_Videdit_lab\data\raw\Videos\Twirl.mp4")
    metadata = extract_metadata(clip)
    validate_metadata(metadata)

    return VideoInput(
        path=path,
        capture = clip,
        fps = metadata["fps"],
        frame_count = metadata["frame_count"],
        duration = metadata["duration"],
        width = metadata["width"],
        height = metadata["height"]

    )