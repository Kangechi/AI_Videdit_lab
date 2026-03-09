"""
Here I want:
To creat a whole input MVP layout design and functionality
A side bar - utilising TKINTER frames
A main panel that will have the canvas - scrollable plus the cards that show the video clips stored
Preview panel still in the main area
- Then unlike working with assets and sessions, we'll haveone constant database that each time you restore
or run the program, it will load the clips
That way we'll have an asset.json - a system memory rather than working memory

"""
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import cv2 as cv
import os
import json
import hashlib
from json import JSONDecodeError
from pathlib import Path

#Ensures: Unique path identifier to also prevnt duplicates

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()



def thumb_gen(video_path, size=(200,200)):
    clip = cv.VideoCapture(video_path)
    ret, frame = clip.read()
    clip.release()
    if not ret:
        return None
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = image.resize(size)
    return ImageTk.PhotoImage(image)



class AssetRegistry:
    def __init__(self, file: str | os.PathLike | None = None):
        # Keep the registry file stable no matter where you launch from (cwd can change).
        # Default: workspace/project root next to your repo-level `assets.json`.
        if file is None:
            project_root = Path(__file__).resolve().parents[2]
            file = project_root / "assets.json"
        self.file = Path(file)
        self.assets = {}
        self.load_assets()
    
    def load_assets(self):
        if self.file.exists():
            try:
                with self.file.open('r', encoding="utf-8") as f:
                    self.assets = json.load(f)
            except JSONDecodeError:
                # If the file is empty/corrupted, start fresh rather than crashing on boot.
                self.assets = {}
                messagebox.showwarning(
                    "assets.json problem",
                    f"Could not parse {self.file.name}. Starting with an empty registry.",
                )
    
    def save_assets(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open('w', encoding="utf-8") as f:
            json.dump(self.assets, f, indent=4)

    def add_asset(self, path: str, metadata: dict):
        if not os.path.exists(path):
            messagebox.showerror("Missing file", "That file no longer exists on disk.")
            return None
        filehash = hash_file(path)
        if filehash in self.assets:
            messagebox.showinfo("Duplicate Video", "Video already uploaded")
            return filehash

        # Persist the path so we can restore cards next launch.
        metadata = dict(metadata or {})
        metadata["path"] = path
        self.assets[filehash] = metadata
        self.save_assets()
        return filehash
    
    def get_assets(self):
        return self.assets
    

class Videdit:
    def __init__(self, root):
        self.root = root
        self.root.title("Videdit 2.0")
        self.root.geometry("1280x1080")
        self.registry = AssetRegistry()

        self.buildui()
        self.load_existing_assets()
    
    def buildui(self):
        self.sidebar = tk.Frame(self.root, width=700)
        self.sidebar.pack(side="left",fill='y', padx = 30, pady =40)

        tk.Button(self.sidebar, text="Home").pack(fill='x', pady=20, padx= 25)
        tk.Button(self.sidebar, text="Uploads").pack(fill='x', pady=20, padx= 25)
        tk.Button(self.sidebar, text="Settings").pack(fill='x', pady=20, padx= 25)
        tk.Button(self.sidebar, text ="Profile").pack(fill='x', pady=20, padx= 25)

        self.main_area = tk.Frame(self.root, width=1000, height=1000, bg="#92A0F7")
        self.main_area.pack(side="right", fill="both", expand=True)

        self.upload_button = tk.Button(self.main_area, text=" + Upload Video",command=self.upload)
        self.upload_button.pack(padx=10, pady=20)

        self.canvas = tk.Canvas(self.main_area, height=350, bg="#5972F7")
        self.canvas.pack(fill='x')

        self.scrollbar = tk.Scrollbar(self.main_area,
                                      orient="horizontal",
                                      bg='#BDF7FF',
                                      command= self.canvas.xview)
        self.scrollbar.pack(fill='x')
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.container = tk.Frame(self.canvas, bg="#B7B5FF")
        self.canvas.create_window((0,0), window=self.container,anchor='nw')
        self.container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion = self.canvas.bbox("all")
            )
        )

    def upload(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]        
            )
        
        if not path:
            return 
        
        filehash = hash_file(path)
        if filehash in self.registry.get_assets():
            messagebox.showinfo("Duplicate Video", "Video already uploaded")
            return
        
        metadata = self.extract_metadata(path)
        if metadata is None:
            return

        self.registry.add_asset(path, metadata)
        self.create_card(path)

    def extract_metadata(self, path):
        clip = cv.VideoCapture(path)
        if not clip.isOpened():
            messagebox.showerror("Error", "Video could not be opened")
            # Common causes: missing codecs/FFmpeg in OpenCV build, corrupt file, or permission issues.
            return None
        
        fps = clip.get(cv.CAP_PROP_FPS)
        frame_count = clip.get(cv.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0 

        width = int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}X{height}"

        clip.release()

        return {
            "duration": duration,
            "fps": fps,
            "resolution": resolution
        }
    def create_card(self, path):
        card = tk.Frame(self.container, width=220,height=220, bd= 2, relief="ridge")
        card.pack(side="left", padx=10, pady=10)
        card.pack_propagate(False)

        thumb = thumb_gen(path)

        if thumb:
            lbl = tk.Label(card, image=thumb)
            lbl.image = thumb
            lbl.pack()

        tk.Label(card, text= os.path.basename(path), wraplength=180).pack()

    def load_existing_assets(self):
        assets = self.registry.get_assets()

        for filehash, metadata in assets.items():
            path = (metadata or {}).get("path")
            if not path:
                continue
            if not os.path.exists(path):
                # Registry remembers clips even if user moved/deleted the file.
                continue
            self.create_card(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = Videdit(root)
    root.mainloop()