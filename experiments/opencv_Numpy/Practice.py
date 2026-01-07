import numpy as np
import cv2 as cv
""""
Errors I made, 
1. forget the self parameter in methods especisally because I'm working with classes
2.Video reading errors - There should always be a loop to read all frames from the video until there are no frames left
3. In function 2, I didn't need to reload the frames list again/ Read the video again
4. In process_video method I didnt convert all frames to gray just a single frame by placing frames 
5. imshow needs to be within a loop, as  I wrote it it only displays a single frame
6. When declaring a function let, path be a parameter so that the function is more flexible and reusable, where the path to the video is passed as an arguemnet
"""
class Video_processing:
    def load_video(self, path):
        frames = []
        clip = cv.VideoCapture(path)
        if not clip.isOpened():
           raise ValueError('Error in opening file')
        
        while True:
         ret, frame = clip.read()
         frames.append(frame)
         if not ret:
            break
        clip.release()
        print(f"Total frames read: {len(frames)}")
        return frames

    def process_video(self, frames):
       gray_frames = [cv.cvtColor(frames, cv.COLOR_BGR2GRAY) for frame in frames ]

       for f in gray_frames:
          cv.imshow('Gray Frame', f)
          cv.waitKey(30) & 0xFF == ord('p')
          cv.destroyAllWindows()
       

processor = Video_processing()
frames = processor.load_video(r'data\raw\Videos\Twirl.mp4')
processor.process_video(frames)

"""
Day 4 - Practice, freelance learning
experimentation
Objectives:
1. Practice Python coding
2. Work with OpenCV library for video processing
3. Numpy arrays and mathematical operations as well as matrix manipulations
4. Numpy and OpenCV integration
"""

