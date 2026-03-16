import cv2 as cv

def shot_boundary_detection(video_path, threshold=20):
    clip = cv.VideoCapture(video_path)
    if not clip.isOpened():
        raise ValueError("Error in opening file")
    fps = clip.get(cv.CAP_PROP_FPS)
    frame_indx = 0
    pre_hist = None
    cuts = []

    while True:
        ret, frame = clip.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv.normalize(hist, hist)

        if pre_hist is not None:
            diff = cv.compareHist(pre_hist, hist,cv.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                cuts.append({"frames": frame_indx,
                             "time": frame_indx/ fps,
                             "scores": float(diff)
                            })
                pre_hist = hist
                frame_indx += 1
                return cuts

video = shot_boundary_detection(r'./data/raw/Videos/Twirl.mp4')
