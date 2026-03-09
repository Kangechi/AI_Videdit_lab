def validate_metadata(metadata: dict) -> None:
    if metadata["duration"] <= 1:
        raise ValueError("Video too short to analyze")
    if metadata["width"] < 320  or metadata["height"] < 240:
        raise ValueError("Video resolution is too low")
    if metadata["fps"] > 120:
        raise ValueError("Video frame rate is unrealistic")
    
