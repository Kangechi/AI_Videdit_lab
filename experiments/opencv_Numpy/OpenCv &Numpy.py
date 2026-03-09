""""
Computers see video as: Frames per second and pixels - the images as numpy array
 Numpy does the algorithmic operations in detecting motion, transitions etc
OPENCV deals with everything with video 

1. Frame creation, 3D array with height, width and channels

"""
import numpy as np
import cv2 as cv
import os

def arrays():
  array_1 = np.array(3)
  array_2 = np.array([3,2,1,5,6,8])
  array_3 = np.array([[3,2,1],[6,5,4]])
  array_4 = np.array([[[3,2,1],[6,5,4],[9,8,7]]])

  print(f"Array 1: {array_1} with dimensions {array_1.ndim}")
  print(f"Array 2: {array_2} with dimensions {array_2.ndim}")
  print(f"Array 3: {array_3} with dimensions {array_3.ndim}")
  print(f"Array 4: {array_4} with dimensions {array_4.ndim}")

  print(f"My array have different shapes: {array_1.shape}, {array_2.shape}, {array_3.shape}, {array_4.shape}")
  
  print(f"Array 2: {array_2[2:5]}")
  print(f"Array 2: {array_2[2:5:2]}")
  print(f"Array 3: {array_3[1,0:2]}")

  video_frame = np.random.randint(0, 255,(10,8,8))
  print(f"Video frame shape: {video_frame}")

  print(f"Video indexing: {video_frame[:, 1:6, 2:5]}")

  brightness_increase = np.clip(video_frame + 50, 0, 255)# increase brightness
  print(f"Brightness increased frame: {brightness_increase}")


def video_step1():
  clip = cv.VideoCapture(r'data\raw\Videos\Twirl.mp4')
  if not clip.isOpened():
   print('Error in opening file')
  else:
    ret, frame = clip.read()
  if ret:
    print(f"Frame shape: {frame.shape}") # height, width, channels
    print(f"Frame data type: {frame.dtype}") # uint8
    print(f"Pixel value at (100,100): {frame[100,100]}") # BGR values
    clip.release()


clip_use = cv.VideoCapture(r'data\raw\Videos\Twirl.mp4')
frame_count = int(clip_use.get(cv.CAP_PROP_FRAME_COUNT))

print(f"Total frames in video: {frame_count}")

frames = []
while True:
  ret, frame = clip_use.read()
  frames.append(frame)
  if not ret:
        break
clip_use.release()
print(f"Total frames read: {len(frames)}")

gray_frames = [cv.cvtColor(f, cv.COLOR_BGR2GRAY)for f in frames if f is not None]

motion_scores = [np.sum(np.abs(gray_frames[i].astype("float") - gray_frames[i-1].astype("float"))) for i in range(1, len(gray_frames))]
downsampled_scores = motion_scores[::10]