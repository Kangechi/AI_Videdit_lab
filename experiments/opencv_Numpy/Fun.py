import numpy as np
import cv2 as cv

"""
Video manipulation using OpenCV and Numpy
"""
class VideoFun:
    """"
    Remember to always have the __init__ method to initialize class attributes as well as self parameters for path, frames and other methods
    """
    def read_display_video(self,path):

        def rescale_video_frames(frame, scale = 0.45):
            width = int(frame.shape[1]* scale)
            height = int(frame.shape[0] * scale)
            dimensions = (width, height)
            return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        
        clip =cv.VideoCapture(path)
        if not clip.isOpened():
            raise ValueError("Error in opening video file")
        frames = []
        while True:
            ret, frame = clip.read()
            """
            Here I rescaled and appended each frame before checking ret value. This was an error. The check for ret should be done first to avoid appending None frames.
            So the correct order:
            if not ret:
            break
            frame = rescale_video_frames(frame)
            frames.append(frame)
            Key Concept: Always validate data before processing to avoid errors.
            The same principle applies with the valueerror that checks the video before the loop starts
            """
            frame = rescale_video_frames(frame)
            frames.append(frame)
            if not ret:
                break
            cv.imshow('Video Frame', frame)
            cv.waitKey(20) & 0xff == ord('p')
            
        clip.release()
        cv.destroyAllWindows()
    
    """
    Data confusion, in frame vs frames
    This ties to the fact that I left out __init__ method and the self.frames attribute that would hold all frames
    Thus all I need to pass was self as a parameter to acces self.frames in the loop
    correct:
    def gray_scale_video(self):
     gray_frames = []
     for frame in self.frames:
       gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
         gray_frames.append(gray)
    """
    def convert_to_grayscale(self,frame):
        gray_frame = []
        for frame in frames:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray_frame.append(gray)
            return 
        while True:
            for gf in gray_frame:
                cv.imshow('Gray Frame', gf)
                cv.waitKey(30) & 0xFF == ord('Q')
        cv.destroyAllWindows()


    def  mask_frame(self, frame):
            """
            Wrong use of masking technique
            the correct way:
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv.circle(mask, (frame.shape[1]//2, frame.shape[0]//2), 40, 255, -1)
            masked = cv.bitwise_and(frame, frame, mask=mask)
            
            """
            edited_frames = frames[500:700, 400:300]
            circle = cv.circle(edited_frames, (frame.shape[1]//2, frame.shape[0]//2), 40, (0,0,0))
            mask = cv.bitwise_and(edited_frames, edited_frames, mask=circle)
            while True:
                for ef in edited_frames:
                    cv.imshow('Edited Frame', ef)
                    cv.waitKey(30) & 0xFF == ord('Q')
            cv.destroyAllWindows()

    def blur_frame(self, frame):
            blurred_frame = frame[:,:,1]
            """
            In slicing you have to consider time and space dimensions
            frames are lists thus the first dimension is time, then height and width
            But pixels are numpy arrays thus the first two dimensions are height and width then channels
            So the correct slicing for blurring all frames would be:        

            
            """
            blur = cv.GaussianBlur(blurred_frame, (5,5))
            return blur
            while True:
                for bf in blurred_frame:
                    cv.imshow('Blurred Frame', bf)
                    cv.waitKey(30) & 0xFF == ord('Q')
            cv.destroyAllWindows()

    def edge_detect(self, frame):
            edges = cv.Canny(frame, 125,500, apertureSize=3)

            return edges
            while True:
                for edf in edges:
                    cv.imshow('Edge Detected Frame', edf)
                    cv.waitKey(30) & 0xFF == ord('Q')
            cv.destroyAllWindows()
            """
            At the end of each function, you can either return the processed frame or display it,
            not both
            
            """


"""
When working with multiple methods in a class, dont nest them as they become local fi=unctions that can't be accessed by the class
"""

class Video_manipulation:
    def __init__(self, path):
        self.path = path
        self.frames = []
    
    def numpy_opperations(self):
        width, height = 800,800
        clip_use = cv.VideoCapture(self.path)
        shape_image = np.zeros([width, height, 3], dtype=np.uint8)
        
        shape_image[:] = [43,45,30]
        shape_image[height//2 -100: height//2 +100, width//2 - 100:width//2 +100] = [40,200,150]
        bitwise_img = cv.bitwise_or(clip_use,clip_use,mask=shape_image)
        """
        Here, clip_use is a VideoCapture object, which cannot be directly used in bitwise operations.
        You need to read the frames and utilise the in the operations
        Correct approach:
        ret, frame = clip_use.read()    
        if ret:
            bitwise_img = cv.bitwise_or(frame, frame, mask=shape_image)     
    
        """