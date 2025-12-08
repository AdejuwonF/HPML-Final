# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import cv2
from collections import deque

import time
prev_frame_time = 0
cur_frame_time = 0
frame_time_deque = deque(maxlen=30) 

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cur_frame_time = time.time()
    # Add Processing Code Here
    fps = 1 / (cur_frame_time - prev_frame_time)
    prev_frame_time = cur_frame_time
    
    frame_time_deque.append(fps)
    fps_text = f"FPS: {int(sum(frame_time_deque) / len(frame_time_deque))}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(5)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
