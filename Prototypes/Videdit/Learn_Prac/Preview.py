import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import cv2 as cv
from PIL import Image, ImageTk
import os
"""
Errors I made:
1. messagebox is a function I needed to import alondside filedialog
2. Validation of file path was missing
- Intially my code was like this:
def upload_video():
    file_path = filedialog.askopenfile(title="select video file, filetype=[("Video File, "*.mp4; *.avi; *.mov)])
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File not found: {file_path}")
        return None - This return none was the error I was experiencing as the none is then taken as the file path and passed to preview_video which then throws an error as it cannot read a video from a None path
    return file_path
Aftward I added a while loop:
while True:
    file_path = filedialog.askopenfile(title="select video file, filetype=[("Video File, "*.mp4; *.avi; *.mov)])
    if not file_path:
        messagebox.showerror("Error", "No file selected")
        return None
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File not found: {file_path}")   
        return None - This was the error I was experiencing as the none is then taken as the file path and passed to preview_video which then throws an error as it cannot read a video from a None path
    return file_path
    -Instead of none before the final returnfile_path instruction, it should have been continue

3. I was using filedialog.askopenfile instead of filedialog.askopenfilename which returns a string path instead of a file object, which is what I needed for the rest of the code to work
4. Last section for running the main loop:
- Validation was also required here as if the user cancels the file dialog, the file variable is None and then passed to preview_video which throws an error as it cannot read a video from a None path
- I added an if statement to check if file is not None before calling preview_video and starting the main loop, and added an else statement to show an error message if no file was selected, and the window will close after that as there is no main loop running

The cap.set(cv.CAP_PROP_POS_FRAMES, frame_no) 
- Sets the current position of the video to a specified frame no- enabling a preview thus human in the loop 
-enables a preview of the video at a specific frame, allowing the user to see a snapshot of the video content before making any edits or decisions. This is particularly useful for video editing applications where users may want to quickly glance at a specific moment in the video without having to play through it entirely.
- Then the frame is converted to RGB format -> PIL Image -> ImageTk.PhotoImage for display
"""

def upload_video():
    while True:
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video File", "*.mp4; *.avi; *.mov")])
        if not file_path:  # User cancelled
            messagebox.showerror("Error", "No file selected")
            return None
        if not os.path.exists(file_path):  # File doesn't exist, ask again
            messagebox.showerror("Error", f"File not found: {file_path}")
            continue
        return file_path
    

def preview_video(file_path, frame_no = any):
    clip = cv.VideoCapture(file_path)
    clip.set(cv.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = clip.read()
    clip.release()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        return ImageTk.PhotoImage(img)
    return None

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Preview")
    file = upload_video()
    if file:
        Frame_img = preview_video(file, frame_no=2)
        if Frame_img:
            label = tk.Label(root, image=Frame_img)
            label.pack()
            root.mainloop()
        else:
            messagebox.showerror("Error", "Could not read video frame")
    # Window closes if no file selected
