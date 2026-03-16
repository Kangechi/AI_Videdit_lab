"""
An integartion of vibe coding with assets and classes file handling and backend development.

In the systems infrastructure - architecture especially when designing the backend design:
1. Media - The user intent to upload
2. Assest - form of memory for the system
- Here we also deal with the assest registry and incoporation of the project
3. Session management

These are the things you have to think about
"""
from PIL import ImageTk
from PIL import Image

import hashlib
def generate_filehash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

import json
import os
import cv2 as cv
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox

def generate_thumbnail(video_path, size=(200, 200)):
    clip = cv.VideoCapture(video_path)
    ret, frame = clip.read()
    clip.release()

    if not ret:
        return None
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = image.resize(size)
    return ImageTk.PhotoImage(image)

class Asset:
    def __init__(self, asset_type, source):
        self.id = str(uuid.uuid4())
        self.asset_type = asset_type
        self.source = source
        self.metadata ={}
        self.state = {}
        self.editable = True

    def update_state(self,key, value):
        self.state[key] = value

    def __repr__(self):
        return f"<Asset Type {self.asset_type}: {self.id[:8]}>"
    

class VideoAsset(Asset):
    def __init__(self, filepath):
        super().__init__(asset_type="video", source= filepath)
        self.filehash = generate_filehash(filepath)
        clip = cv.VideoCapture(filepath)
        if not clip.isOpened():
            raise ValueError("Video Cannot open")
        
        frame_count = clip.get(cv.CAP_PROP_FRAME_COUNT)
        fps = clip.get(cv.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0

        width = int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))

        clip.release()

        self.metadata = {
            "duration": duration,
            "fps": fps,
            "resolution": (width, height)
        }
        self.state = {
            "trim_start": 0,
            "trim_end": duration
        }

class AssetRegistry:
    def __init__(self):
        self.assests = {}
        self.hash_index = {}
    
    def add_asset(self, asset):
        if asset.source in self.hash_index:
            raise ValueError("Video already uploaded")
        
        
        self.assests[asset.id] = asset
        self.hash_index[asset.filehash] = asset.id
    
    def get_asset(self,asset_id):
        return self.assests.get(asset_id)
    
    def list_assets(self):
        return list(self.assests.values())
    

class SessionManager:
    def __init__(self, registry_file= "video_session.json"):
        self.registry_file = registry_file
        self.registry = self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def create_session(self, asset):
        session_id = asset.id

        if session_id in self.registry:
            raise ValueError("Session ID exists")
        
        self.registry[session_id] = {
            "video_path": asset.source,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "uploaded",
            "metadata": asset.metadata

        }

        self.save_registry()


class Videdit:
    def __init__(self, root):
        self.root = root
        self.root.title("Videdit")

        self.asset_reg = AssetRegistry()
        self.session_man = SessionManager()

        self.build_ui()

    def build_ui(self):
        self.upload_button = tk.Button(self.root,
                                       text="Upload Video",
                                       command=self.upload_video,
                                       height=2)
        self.upload_button.pack(pady=10)
        
        self.canvas = tk.Canvas(self.root, height=250)
        self.canvas.pack(fill="x")
        self.scrollbar = tk.Scrollbar(
            self.root,
            orient="horizontal",
            command= self.canvas.xview

        )
        self.scrollbar.pack(fill="x")
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.container = tk.Frame(self.canvas)
        self.canvas.create_window((0,0), window= self.container, anchor="nw")
        self.container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

    def upload_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File",
                                               filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        
        # check for duplicates first
        for asset in self.asset_reg.list_assets():
            if asset.source == file_path:
                messagebox.showerror("Error", "Video already uploaded")
                return
        
        # create and register new video asset
        try:
            video = VideoAsset(file_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.asset_reg.add_asset(video)
        self.session_man.create_session(video)
        self.create_card(video)

    def create_card(self, asset):
        card = tk.Frame(self.container,
                        width = 220,
                        height = 220,
                        bd=2, relief="ridge")
        
        card.pack(side="left", padx=10, pady =10)
        card.pack_propagate(False)
        thumbnail = generate_thumbnail(asset.source)
        if thumbnail:
            img_label = tk.Label(card, image=thumbnail)
            img_label.image = thumbnail
            img_label.pack()
        
        name_label = tk.Label(
            card,
            text = os.path.basename(asset.source),
            wraplength=180
        )
        name_label.pack()
        btn_frame = tk.Frame(card)
        btn_frame.pack(pady=5)


       

        preview_btn = tk.Button(btn_frame,
                                text="Preview",
                                command=lambda: self.preview(asset))
        preview_btn.pack(side="left",padx=5)

    def preview(self, asset):
        clip = cv.VideoCapture(asset.source)
        if not clip.isOpened():
            messagebox.showerror("Error","Error in opening video")
        
        while clip.isOpened():
            ret, frame = clip.read()
            if not ret:
                break
            resized_frame = cv.resize(frame, (800,800))
            cv.imshow("Preview", resized_frame)
            if cv.waitKey(30) & 0xff  == ord('q'):
                break

        clip.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = Videdit(root)

    root.mainloop()