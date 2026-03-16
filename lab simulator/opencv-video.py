import cv2 as cv

frames = []
clip = cv.VideoCapture(r"data\raw\Videos\Twirl.mp4")
if not clip.isOpened():
    raise ValueError("Video file not opened")

while True:
    ret, frame = clip.read()
    if not ret:
        break
    frames.append(frame)
    clip.release()
    
    cv.imshow("Twirl",frame)
    if cv.waitKey(60) & 0xff == ord("q"):
        break
   
    cv.destroyAllWindows()
