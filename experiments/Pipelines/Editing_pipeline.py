import cv2 as cv
import numpy as np

def load_extract(video_path):
    """
    You don't need to define a function to resize:
    You can:
    1. Declare variable for height, width and ensure you convert them to int
    width = int(frames.shape[1] * scale)
    height = int(frames.shape[0] * scale)
    then frame = cv.resize(frame, (width, height), interpolation )
     NOTE: Functions do one thing then return data thus at the end of the code instead of cv.imshow it should ne return
    """
    #Unecessary function
    def resize_vid(frame, scale = 0.45):
       width, height = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
       dimensions = (width, height)
       return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    
    clip = cv.VideoCapture(video_path)
    if not clip.isOpened():
        raise ValueError("Error in opening video file")
    
    frames = []
    while True:
     ret, frame = clip.read()
     if not ret:
        break
     frame = resize_vid(frame)
     frames.append(frame)
     cv.imshow("Video", frame)

    clip.release()

def blur(frames):
   blurred_frames = []
   for frame in frames:
      blurred = cv.GaussianBlur(frame, (5,5), 0)
      blurred_frames.append(blurred)
      cv.imshow("Blurred video", blurred)
      if cv.waitKey(30) & 0xff == ord('p'):
         break
      
def edge_detection(frames):
   for frame in frames:
      edge = cv.Canny(frame, 120,255)
      cv.imshow("Edges", edge)
      if cv.waitKey(30) & 0xff == ord('p'):
         break
      """
      Here and for most of the other functions, the parameter is the frames
      thus eg.
     gray_frames = []
     return [cv.color(f, cv.colo_bgr2gray) for f in frames
     gray_frames.append(f)
     ]
      
     blur and canny the same then for canny ensure you work with grayscale frames
     return [cv.gaussianblur(f, (5,5), 0)]
     

     edges = []
     for g in gray_frames:
      edges.append(cv.canny(g, 125,255))
     return edges
      """
def gray_video(frames):
   gray_frames = []
   for frame in frames:
      gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      gray_frames.append(gray_frame)
      cv.imshow("Gray_scale", gray_frame)
      if cv.waitKey(20) & 0xff == ord('p'):
         break


"""
For masking:
1. Start with the definition of the shape of the mask
h, w = frames[0],shap[:2]
mask = np.zeros((h,w), dtype = np.uint8)
cv.circle(mask, (h//2, w//2), 40, 255, 1 )
masked_frames = []
for frame in frames:
masked = cv.bitwise_and(frame, frame, mask=mask)
masked_frames.append(masked)
return masked_frames
"""
def mask_video(frames):
   mask = np.zeros(frame.shape[:2],dtype = "unit8")
   mask_use = cv.circle(mask, (frame.shape[1]// 2, frame.shape[0 // 2]), 40, 255,1) 
   for frame in frames:
      masked =  cv.bitwise_and(frame, frame, mask=mask_use)

      cv.imshow("Masked",masked)
      if cv.waitKey(20) & 0xff == ord('p'):
         break
      cv.release()
      cv.destroyAllWindows()


"""
Just like edge detction, motion score work with gray_frames
for i in range(len(gray_frames) - 1):
        diff = cv.absdiff(gray_frames[i], gray_frames[i+1])
        motion_scores.append(np.mean(diff))
    return motion_scores
"""
def motion_score(frames):
   motion_score = []
   for i in range(len(frames) - 1):
      diff = cv.absdiff(frames[i], frames[i+1])
      gray = gray_video(diff)
      motion = cv.mean(gray)
      motion_score.append(motion)

      print(f"Frame{i} -> {frames{i+1}} motion: {motion_score: .2f}")

      for i, frame in enumerate(frames[: -1]):
         display = frame.copy()

         cv.putText(display, f"Motion scores: {motion_score[1]: .2f}",[20,40],cv.FONT_HERSHEY_COMPLEX, 1.0)
         cv.imshow("Text_overlay", display)
         if cv.waitKey(20) & 0xff == ord('p'):
            break
               

