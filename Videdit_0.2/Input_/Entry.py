"""
Docstring for Videdit_0.2.Input_.Entry
For here I want to create a class with GUI
for the entry
It will incorporate:
 input
   |
Validation
   |
Registartion
   |
Preview

-This is my first phase of building
"""
import os
import cv2 as cv
from tkinter import *
from tkinter import filedialog, Tk
import json



class Video_input_layer:
   def __init__(self, register_file = "video_registry.json"):
      self.registry_file = register_file
      self.registry = self.load_registry()
   
   def load_registry(self):
      if os.path.exists(self.registry_file):
         with open(self.registry_file, 'r') as f:
            return json.load(f)
      return {}
   
   def save_registry(self):
      with open(self.registry_file, "w") as f:
         json.dump(self.registry, f, indent=4)
   
   def select_file(self):
      root = Tk()
      root.withdraw()
      filepath = filedialog.askopenfilename(type="Select Video file", filetypes=[("Video Files", "*.mp4 *.mov; *.avi")])

      return filepath
   
   def validate_file(self, filepath):
      if not filepath or not os.path.exists(filepath):
         print("File not found")
         return False
      filename = os.path.basename(filepath)
      if filename in self.registry:
         print("File already uploaded")
         return False
      
      cap = cv.VideoCapture(filepath)
      if not cap.isOpened():
         print("File cannot open")
         return False
      
      frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
      fps = cap.get(cv.CAP_PROP_FPS)
      duration = frame_count / fps
      width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
      
      if duration > 600:
         print("Video too long")
         return False
      
      cap.release()
      return {
         "filename": filename, 
         "path": filepath,
         "duration": duration,
         "resolution": (width, height)
      }
   def store_video(self, metadata):
      self.registry[metadata["filename"]] = metadata
      self.save_registry()
      print(f"Video {metadata["filename"]} stored successfully")
   
   def preview_video(self, file_path):

        # OpenCV window to preview video
      cap = cv.VideoCapture(file_path)
      if not cap.isOpened():
            print("Cannot open video.")
            return

      print("Press 'q' to quit preview.")
      while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv.resize(frame, (640, 500))
            cv.imshow("Preview", resized_frame)
            if cv.waitKey(30) & 0xFF == ord("q"):
                break

      cap.release()
      cv.destroyAllWindows()
      
   

video = Video_input_layer()
path = video.select_file()
video.save_registry()
video.store_video(video.validate_file(path))
video.load_registry()
video.preview_video(path)