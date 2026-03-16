import uuid
import tkinter as tk
from tkinter import *
from tkinter import filedialog

class Asset:
    def __init__(self, asset_type, source):
        self.id = str(uuid.uuid4())
        self.asset_type = asset_type
        self.source = source
        self.metadata = {}
        self.state = {}
        self.editable = True
    
    def update_state(self, key, value):
        self.state[key] = value
    
    def __repr__(self):
        return f"<Asset {self.asset_type} | {self.id[:8]}>"
    

class VideoAsset(Asset):
    def __init__(self,filepath,duration, fps):
        super().__init__(asset_type="video", source= filepath)
        self.metadata = {
            "duration": duration,
            "fps": fps 
          }
        self.state = {
            "trim_start": 0,
            "trim_end": "duration"
        }

class AssetRegistry:
    def __init__(self):
        self.assets = {}

    def add_asset(self, asset):
        self.assets[asset.id] = asset

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)
    
    def list_assets(self):
        return list(self.assets.values())
    
class Videdit:
    def __init__(self, root):
        self.root = root
        self.root.title("Asset Manager - Videdit")
        self.registry = AssetRegistry()

        #UI

        self.asset_listbox = tk.Listbox(root, width=60)
        self.asset_listbox.pack(side=LEFT, fill=BOTH, expand=True)

        control_frame = tk.Frame(root) 
        control_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.upload_button = tk.Button(control_frame, text="Upload Asset", command=self.upload_video)
        self.upload_button.pack(pady=15)

    def upload_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])

        if not file_path:
            return
        
        video_asset = VideoAsset(file_path, duration=None, fps=None)
        self.registry.add_asset(video_asset)
        self.asset_listbox.insert(tk.END, repr(video_asset))


if __name__ == "__main__":
    root = tk.Tk()
    app = Videdit(root)
    root.mainloop()