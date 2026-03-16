When I chose to build a video editing software, the easy part was visualizing its workflows, design and how it would be uesed by millions.
However, te part I didnt anticipate was learning to work with systems, the computer language and understanding of video data.

In terms of systems:
 - It was about understanding the workflows
          upload video
               |
          Verification
               |
        Metadata_extraction
               |
        Frame_extraction
               |
        Shot-boundary detection - for the computer to understand the different scenes.
- This occurs through an algorithmic breakdown of DAD
              D- Difference
              A-Aggregation
              D - Decisions (Through thresholds)

 A snippet example:
 """
 Using histogram difference to detect whether a shot has taken place
 """
  def shot_detection(frames: list(frame_idx, frame) -> list[dict], thresh= 0.5):
          boundaries = []
          prev_hist = None

          for frame_idx, frame in frames:
          gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
          """
          Convert to grayscale for ease in computation

          """
          hist = cv.calcHist([gray], [0], None, [256], [0,256])
          cv.Normalize(hist, hist)
          """
          Normalize to ensure the output matches a certain scale
          """

          if prev_hist is not None:
              diff =cv.compareHist(prev_hist, hist, cv.HISTCMP_BHATTACHYRRA)
              if diff > thresh:
                 boundaries.append({
                    "frame:" frame_idx,
                    "diff_score": round(diff, 4),
                    "type": "hard-cut" if diff > 0.7 else "dissolve",
                    "confidence": min(diff/ 1.0, 1.0)
                    
                         })
        prev_hist = hist

        return boundaries

- So this is how the system detect change, in scenes. through calculating the histogram difference of the two frame in grayscale. soring the boundaries later for scene detection dring template generation.
                       |
            Motion detection

- Here we us an opticalflow to detect the motion scores within a video:
def calculation_motion(frames: list(frame_idx, frame)):
    scores = []
    prev_gray = None

    for frame_idx, frame in frames:
        gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if  prev_gray is not None:
           flow = cv.opticalFlowFarneback(prev_gray, gray, None,
           """
           The difference of prev frame and current frame - then certain thresholds to help in motion detection: hence the use of the pyr_scale/level etc
           """
           pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
           )
        magnitude, angle = cv.cartToPolar(flow[...,0], flow[..., 1])
        mean_mag = float(np.mean(magnitute))
        dominant_angle = float(np.degrees(np.mean(angke)))

         scores.append({
                "frame":        frame_idx,
                "motion_score": round(mean_mag, 4),
                "motion_type":  classify_motion(mean_mag, dominant_angle),
                "flow_angle":   round(dominant_angle, 2)
            })
        prev_gray = gray

    return scores

def classify_motion(score:float, angle):
       if score< 0.5 : return "static"
       if score < 2.0:  return "handheld"
       if 80 < angle < 100: return "tilt"
        if angle < 20 or angle > 340: return "pan"
    return "high_motion" 
                   |
            Visual display extraction
    - Here you extract the colour pallets and the dominant colors of the video; It utilise K-clusters an ML model in this case to cluster pixels of the same color

import sklearn,clusters import KMeans
def generate_palette(frame: np.ndarray, n_colors: int = 5 -basically the no of colors/clusters):
      """
      Key things 1. Reshape the frame inorder to just have pixels
      2. Sample in order to ease computation
      """
      pixels = frame.shape(-1, 3)astype(np.float32)
      sample = pixels[np.random.choice(len(pixels)), min(3000, len(pixels)), replace=False]
      kmeans = Kmeans(n_clusters = n_colors, n_init=5, random_state=0).fit(sample)

      palette=[]
      for center in kmeans.cluster_centers_:
        b, g, r = [int(c) for c in center]  # OpenCV is BGR
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        palette.append({"hex": hex_color, "rgb": (r, g, b)})
    return palette

def analyze_visual_features(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "brightness":  round(float(np.mean(gray)) / 255, 3),
        "contrast":    round(float(np.std(gray)) / 255, 3),
        "saturation":  round(float(np.mean(hsv[:,:,1])) / 255, 3),
        "palette":     extract_color_palette(frame),
    }

                     |
        Object _Detection
- Here we use haar cascades with a gray image to identify faces, smiles eyes etc
- YOLO MODEL - detect object using an rgb image 
- The detection of these object enable: safe teaxt araes for captions to be detected and created

from ultralytics  import YOLO
face_cascade  = cv.CascadeClassifier(cv.data.haarcascade + "haarcascade_frontalface_default.xml")
yolo_model = YOLO(""yolov8n.pt"")
def object_detect(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    faces  = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results  = yolo_model(rgb, verbose=False)[0]
    objects  = [
        {
            "class":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox":       [round(v) for v in box.xyxy[0].tolist()],
        }
        for box in results.boxes
        if float(box.conf) > 0.4
    ]

    return {
        "faces":   [{"x":int(x),"y":int(y),"w":int(w),"h":int(h)} for x,y,w,h in faces],
        "objects": objects,
        "safe_zone": compute_safe_zone(frame.shape, faces, objects),
    }

def compute_safe_zone(shape, faces, objects) -> dict:
    """Returns the largest rect with no detected subjects — safe for text."""
    h, w = shape[:2]
    occupied_y = max([f[1]+f[3] for f in faces], default=0)
    return {"x": 0, "y": occupied_y + 20, "w": w, "h": h - occupied_y - 20}
