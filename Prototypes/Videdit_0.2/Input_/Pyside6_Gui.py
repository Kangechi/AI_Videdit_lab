"""
PySide6 version of the Videdit Input GUI.
Sidebar + upload button + horizontally scrollable video cards,
with persistent `assets.json` registry.
"""
import sys
import os
import json
import hashlib
from pathlib import Path
from json import JSONDecodeError
import cv2 as cv
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QScrollArea,
)
def hash_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def first_frame_pixmap(video_path: str, size=(200, 200)) -> QPixmap | None:
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    bytes_per_line = 3 * w
    img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(img)

    return pix.scaled(
        size[0],
        size[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class AssetRegistry:
    def __init__(self, file: str | os.PathLike | None = None):
        # Keep the registry file stable no matter where you launch from.
        if file is None:
            project_root = Path(__file__).resolve().parents[2]
            file = project_root / "assets.json"
        self.file = Path(file)
        self.assets: dict[str, dict] = {}
        self.load_assets()
    def load_assets(self) -> None:
        if not self.file.exists():
            self.assets = {}
            return
        try:
            with self.file.open("r", encoding="utf-8") as f:
                self.assets = json.load(f)
        except JSONDecodeError:
            self.assets = {}

    def save_assets(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("w", encoding="utf-8") as f:
            json.dump(self.assets, f, indent=4)

    def add_asset(self, path: str, metadata: dict) -> str | None:
        if not os.path.exists(path):
            return None
        filehash = hash_file(path)
        if filehash in self.assets:
            return filehash
        metadata = dict(metadata or {})
        metadata["path"] = path
        self.assets[filehash] = metadata
        self.save_assets()
        return filehash
    
    def get_assets(self) -> dict[str, dict]:
        return self.assets
    
class AssetCard(QFrame):
    def __init__(self, video_path: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(220, 240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = first_frame_pixmap(video_path, size=(200, 200))
        if pix is not None:
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setText("No preview")
        name = QLabel(os.path.basename(video_path))
        name.setWordWrap(True)
        layout.addWidget(thumb_label)
        layout.addWidget(name)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Videdit 2.0 (PySide6)")
        self.resize(1280, 720)
        self.registry = AssetRegistry()
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(170)
        sidebar.setStyleSheet("background: #A1AEFF;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        for text in ["Home", "Uploads", "Settings", "Profile"]:
            side_layout.addWidget(QPushButton(text))
        side_layout.addStretch(1)
        # Main area
        main = QFrame()
        main.setStyleSheet("background: #92A0F7;")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        upload_btn = QPushButton("+ Upload Video")
        upload_btn.clicked.connect(self.upload_video)
        main_layout.addWidget(upload_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        # Scrollable horizontal cards
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(8, 8, 8, 8)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards_container)
        main_layout.addWidget(self.scroll)
        root_layout.addWidget(sidebar)
        root_layout.addWidget(main, stretch=1)
        self.load_existing_assets()

    def extract_metadata(self, path: str) -> dict | None:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"duration": duration, "fps": fps, "resolution": f"{width}x{height}"}
    
    def add_card(self, path: str) -> None:
        stretch_index = self.cards_layout.count() - 1
        self.cards_layout.insertWidget(stretch_index, AssetCard(path))
    def upload_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov)",
        )
        if not path:
            return
        filehash = hash_file(path)
        if filehash in self.registry.get_assets():
            QMessageBox.information(self, "Duplicate Video", "Video already uploaded")
            return
        metadata = self.extract_metadata(path)
        if metadata is None:
            QMessageBox.critical(
                self,
                "Error",
                "Video could not be opened (codec/permission/corrupt file).",
            )
            return
        self.registry.add_asset(path, metadata)
        self.add_card(path)

    def load_existing_assets(self) -> None:
        for _, meta in self.registry.get_assets().items():
            path = (meta or {}).get("path")
            if path and os.path.exists(path):
                self.add_card(path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
