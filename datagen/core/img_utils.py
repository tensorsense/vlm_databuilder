from pathlib import Path
from moviepy.editor import VideoFileClip
from typing import Optional
import cv2 #, numpy as np

def cv2_resize(img, maxwidth, maxheight):
    f1 = maxwidth / img.shape[1]
    f2 = maxheight / img.shape[0]
    f = min(f1, f2)  # resizing factor
    dim = (int(img.shape[1] * f), int(img.shape[0] * f))
    resized = cv2.resize(img, dim)
    return resized

def get_frames(video: str | VideoFileClip, start_frame=0, end_frame: Optional[int]=None, nframes=8):
    if type(video) is str:
        video = VideoFileClip(video)
    start_time = start_frame / video.fps
    end_time = end_frame / video.fps if end_frame is not None else video.duration
    frames = [video.get_frame((i+1)/(nframes+1)*(end_time-start_time) + start_time) for i in range(nframes)]
    video.close()
    return frames