import cv2 as cv
import matplotlib.pyplot as plt


def shot_detection(video_path, threshold=0.65):
    clip = cv.VideoCapture(video_path)
    if not clip.isOpened():
        raise ValueError("Error in opening video")
    
    prev_hist = None
    shots = []
    frame_count = 0

    while True:
        ret, frame = clip.read()
        if not ret:
            break
        frame_count += 1

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0,1,2], None, [50, 60, 50], [0,180,0,256,0, 256])
        cv.normalize(hist,hist)

        if prev_hist is not None:
            diff = cv.compareHist(prev_hist, hist, cv.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                shots.append({"frame": frame_count, "score": float(diff)})
                print(f"Shot detected at frame {frame_count} with score {diff:.4f}")
            prev_hist = hist
           
    
    return shots


cap = shot_detection(r"C:\Users\ADMIN\Desktop\AI_Videdit_lab\data\raw\Videos\Twirl.mp4")
