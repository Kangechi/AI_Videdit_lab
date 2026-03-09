import cv2 as cv
import numpy as np

def video_slicing(path):
    def re_scaler(frame,scaler= 0.45):
        width = int(frame.shape[1] * scaler)
        height = int(frame.shape[0] * scaler)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

  
    clip = cv.VideoCapture(path)
    if not clip.isOpened():
        return  ValueError("Error in opening file")
    
    frames = []
    while True:
        ret, frame = clip.read()
        if not ret:
            break
        resize_frames = re_scaler(frame)
        frames.append(resize_frames)
    clip.release()
    print(f"Frames extracted {len(frames)}")

    
    """
    Slicing
    """
    intro = frames[:30]
    middle = frames[70:250]
    end = frames[-30:]

    print(f"Intro frames {len(intro)}")
    print(f"Main frames {len(middle)}")
    print(f"Outro frames {len(end)}")


    """
    Just playing one frame of the intro, main and outro.
    Time slicing as the frames read the 346 have been stored in the frames list thus the time stamps can be 
    sliced out of the main frames list.
    To play the video, you have to loop through the sequence of frames
"""
   
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    for frame in middle:
        cv.imshow("main", frame)
        if  cv.waitKey(20) & 0xff == ord('p'):
            break
    
def Motion_detection(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        return ValueError("Error accessing file")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    motion_scores = []

    for i in range(len(frames) - 1):
        diff = cv.absdiff(frames[i], frames[i+1])
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        motion_score = np.mean(gray)
        motion_scores.append(motion_score)

        print(f"Frame{i} -> {i+1}: Motion score = {motion_score: .2f}" )

    for i, frame in enumerate(frames[:-1]):
        display = frame.copy()

        cv.putText(display, f"Motion: {motion_scores[1]:.2f}", (20,40),cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(40, 56,60), 1)

        cv.imshow("Display", display)
        if cv.waitKey(20) & 0xff == ord('p'):
            break
        def shot_bound(motion_scores, threshold=15):
            shot_boundaries = []

            for i, motion_score in enumerate(motion_scores):
                if motion_score > threshold:
                    shot_bound.append(i)
            
                return shot_boundaries


def Masking(path):
    clip_2 = cv.VideoCapture(path)
    if not clip_2.isOpened():
        return ValueError("Error in opening file")
    frames = []

    while True:
        ret, frame = clip_2.read()
        if not ret:
            break
        frames.append(frame)
    clip_2.release()

    masked_frames =  []
    for frame in frames:
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[200:500, 300:600] = 255
        focused = cv.bitwise_and(frame, frame, mask=mask)
        masked_frames.append(focused)

        cv.imshow("Mask", focused)
        if cv.waitKey(30) & 0xff == ord('p'):
            break
    cv.destroyAllWindows()




video = Masking(r'./data/raw/Video/Twirl.mp4')

