#Practicing using dictionaries in Python - To understand data structure
#Data Structure - the input and output of the video data
video_metadata = {
    "video": "Kiboi.mp4",
    "frames": 30,
    "duration": 300,
    "audio": True,
    "resolution": "1920x1080"
}
#Alternative
video_metadata_alt = dict(video ="Kiboi.mp4", frames = 30)
video_metadata_alt2 = dict([("video", "Kiboi.mp4"),("frames", 30)])

video_metadata["audio"] = False  #Updating value
video_metadata_alt["crop"] = True
video_metadata_alt["frames"] = 90

print(video_metadata_alt.keys())
print(video_metadata.values())

#Frame loops
#Looping through the frames that make the video

for frame in range(video_metadata["frames"]):
    if frame % 10 == {1,2,3}:
        print("Even frame detected")


#Pipeline = to function stages
#Example
def load_video(video_file):
    print(f"Video loading: {video_file}")
    return video_file

def process_video(video_file):
    print(f"Processing video: {video_file}")
    return video_file


def extract_frames(video_file, num_frames):
    print(f"Extracting {num_frames} frames from {video_file}")

#Pipelines represent the workflow of video processing